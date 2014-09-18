#ifndef SCPHD_CUH
#define SCPHD_CUH

#include <vector>
#include <ctime>
#include "kernels.cuh"
#include "types.h"
#include "gmreduction.h"
#include "math.h"

#define DEBUG_MSG(x) if ( verbosity_ >= 2) cout << "[" << __func__ << "(" << __LINE__ << ")]: " << x << endl
#define DEBUG_VAL(x) if ( verbosity_ >= 2) cout << "[" << __func__ << "(" << __LINE__ << ")]: " << #x << " = " << x << endl

using namespace std ;

template <class G>
bool weightBelow(G gaussian){
    return (gaussian.weight < 0.1) ;
}

/// helper function for outputting a Gaussian to std_out
template<int N,int N2=N*N>
__host__  void
print_feature(Gaussian<N,N2> f)
{
    int dims = N ;
    cout << "dims: " << dims << endl ;
    cout << "weight: " << f.weight << endl ;
    cout << "mean: " << endl ;
    for ( int i = 0 ; i < dims ; i++ )
        cout << f.mean[i] << " " ;
    cout << endl << "cov: " << endl ;
    for ( int i = 0 ; i < dims ; i++){
        for ( int j = 0 ;j < dims ; j++)
            cout << f.cov[i+j*dims] << " " ;
        cout << endl ;
    }
    cout << endl ;
//#endif
}

template <int N, class G, int M, class ParentMM, class DaughterMM, class MeasureModel>
class SCPHDFilter{
public:
    vector< vector<double> > parent ;
    vector< vector<G> > daughter ;
    vector<double> parentWeights ;
    int parentDim ;
    int nParticles ;
    int nDaughter ;
    int nMeasure ;
    double w0 ;
    double kappa ;
    double ps ;
    double pd ;
    double pruneWeight ;
    double mergeDist ;
    double minNeff ;

    ParentMM parentMotionModel ;
    DaughterMM daughterMotionModel ;
    MeasureModel measureModel ;

    SCPHDFilter(int nParticles_, double ps_, double pd_, double w0_, double kappa_,
                double pruneWeight_, double mergeDist_,
                ParentMM parentMotionModel_, DaughterMM daughterMotionModel_,
                MeasureModel measureModel_){

        nParticles = nParticles_ ;
        ps = ps_ ; pd = pd_ ; w0 = w0_ ; kappa = kappa_ ;
        pruneWeight = pruneWeight_ ; mergeDist = mergeDist_ ;
        parentMotionModel = parentMotionModel_ ;
        daughterMotionModel = daughterMotionModel_ ;
        measureModel = measureModel_ ;
        minNeff = 0.5 ;
        verbosity_ = 3 ;



        initializeParent(nParticles);
        initializeDaughter(nParticles) ;
        offsets.assign(nParticles+1,0);
    }


    ~SCPHDFilter(){
        cudaFree((void*)dParent) ;
        cudaFree((void*)dDaughter) ;
        cudaFree((void*)dMeasurements) ;
    }

    void predict(){
//        cout << "predictParent" << endl ;
         predictParent() ;

//        cout << "computeOffsets" << endl ;
        computeOffsets() ;

//        cout << "copyDaughter" << endl ;
        copyDaughter() ;

        int gridDim = nParticles ;
        int blockDim = 256 ;
//        cout << "daughterPredictKernel" << endl ;
        daughterPredictKernel<G,DaughterMM,N><<<gridDim,blockDim>>>(dDaughter,dParent,
                                                  daughterMotionModel,
                                                  dOffsets, nParticles) ;
        copyParent();
    }

    void update(vector< vector<double> > measurements){
        copyMeasurements(measurements);
        DEBUG_VAL(nMeasure) ;
        dim3 gridDim ;
        dim3 blockDim ;

        // compute births
        G* dBirths ;
        size_t birthSize = sizeof(G)*nParticles*nMeasure ;
        cudaMalloc((void**)&dBirths,birthSize) ;
        gridDim.x = nParticles ;
        blockDim.x = 256 ;

        computeBirthsKernel<G,MeasureModel,N,M><<<gridDim,blockDim>>>
            (dMeasurements,dParent,measureModel,w0,nMeasure,nParticles,
             dBirths) ;

//        // peek births
//        vector<G> births(nParticles*nMeasure) ;
//        cudaMemcpy(&births[0],dBirths,birthSize,cudaMemcpyDeviceToHost) ;
//        for ( int  i = 0 ; i < nMeasure ; i++){
//            print_feature<N>(births[i]) ;
//        }

        // compute detections
        G* dDetections ;
        size_t detectionSize = nMeasure*nDaughter*nParticles*sizeof(G) ;
        cudaMalloc((void**)&dDetections,detectionSize) ;
        blockDim.x = 32 ;
        blockDim.y = 32 ;
        computeDetectionsKernel<G,MeasureModel,N,M><<<gridDim,blockDim>>>
            (dParent,dDaughter,dMeasurements, measureModel, dOffsets,
             nMeasure,nDaughter,dDetections) ;
        cudaDeviceSynchronize() ;

        // compute weight normalizers
        double* dNormalizers = NULL ;
        size_t normalizerSize = nMeasure*nParticles*sizeof(double) ;
        cudaMalloc((void**)dNormalizers,normalizerSize) ;
        gridDim.x = min(nParticles,1024) ;
        gridDim.y = min(nMeasure,32) ;
        blockDim.x = 256 ;
        blockDim.y = 1 ;
        DEBUG_VAL(normalizerSize/sizeof(double)) ;
        DEBUG_VAL(nParticles) ;
        DEBUG_VAL(nMeasure) ;
        DEBUG_VAL(w0) ;
        DEBUG_VAL(kappa) ;
        computeNormalizersKernel<<<gridDim,blockDim>>>
            (dDetections,w0,kappa,dOffsets,nParticles,nMeasure,dNormalizers) ;
        cudaDeviceSynchronize() ;

        // re-weight parent particles
        vector<double> normalizers(nMeasure*nParticles,1.0) ;
//        cudaMemcpy(&normalizers[0],dNormalizers
//                   ,normalizerSize,cudaMemcpyDeviceToHost) ;
        for (int i = 0 ; i < nParticles ; i++){
            for (int m = 0 ; m < nMeasure ; m++){
                if (i == 0){
//                    DEBUG_VAL(normalizers[m]) ;
                }
                parentWeights[i] *= normalizers[nMeasure*i+m] ;
            }
            double cnPredict = 0 ;
            for (int j = 0 ; j < daughter[i].size() ; j++){
                cnPredict += daughter[i][j].weight ;
            }
            parentWeights[i] *= exp(-cnPredict) ;
        }

        // do the update
        G* dUpdate ;
        size_t updateSize = birthSize + detectionSize + nDaughter ;
        cudaMalloc((void**)&dUpdate,updateSize) ;

        gridDim.x = nParticles ;
        gridDim.y = 1 ;
        blockDim.x = 256 ;
        blockDim.y = 1 ;
        updateKernel<<<gridDim,blockDim>>>
            (dDaughter,dDetections,dBirths,dNormalizers, pd,
             dOffsets, nMeasure, dUpdate) ;
        readDaughterUpdate(dUpdate);

        for ( int i = 0 ; i < daughter[0].size() ; i++){
            print_feature(daughter[0][i]) ;
        }
        cout << endl;

        // GM reduction
        for (int n = 0 ; n < nParticles ; n++){
            daughter[n] = reduceGaussianMixture<G>(daughter[n],pruneWeight,mergeDist) ;
        }


        // clean up
        cudaFree(dNormalizers) ;
        cudaFree(dDetections) ;
        cudaFree(dBirths) ;
        cudaFree(dUpdate) ;
    }

    double computeNeff(){
        double sum_of_squares = 0 ;
        for (int i = 0 ; i < nParticles ; i++){
            sum_of_squares += parentWeights[i]*parentWeights[i] ;
        }
        return 1.0/(nParticles*sum_of_squares) ;
    }

    void resample(){
        double nEff = computeNeff() ;
        if (nEff >= minNeff)
            return ;

        vector<int> idxResample(nParticles) ;
        double interval = 1.0/nParticles ;
        double r = randu<double>() * interval ;
        double c = parentWeights[0] ;
        int i = 0 ;

        vector< vector<double> > resampledParent(N) ;
        vector< vector<G> > resampledDaughter(nParticles) ;

        for ( int j = 0 ; j < nParticles ; j++ )
        {
            while( r > c )
            {
                i++ ;
                // sometimes the weights don't exactly add up to 1, so i can run
                // over the indexing bounds. When this happens, find the most highly
                // weighted particle and fill the rest of the new samples with it
                if ( i >= nParticles || i < 0 || isnan(i) )
                {
                    double max_weight = -1 ;
                    int max_idx = -1 ;
                    for ( int k = 0 ; k < nParticles ; k++ )
                    {
                        if ( exp(parentWeights[k]) > max_weight )
                        {
                            max_weight = exp(parentWeights[k]) ;
                            max_idx = k ;
                        }
                    }
                    i = max_idx ;
                    // set c = 2 so that this while loop is never entered again
                    c = 2 ;
                    break ;
                }
                c += parentWeights[i] ;
            }
            idxResample[j] = i ;
            r += interval ;

            for (int n = 0 ; n < N ; n++){
                resampledParent[n].push_back(parent[n][i]) ;
            }
            resampledDaughter.push_back(daughter[j]);
        }
        parent = resampledParent ;
        daughter = resampledDaughter ;
        parentWeights.assign(nParticles,interval);
    }

    vector<double> getParticle(int idx){
        vector<double> particle(N) ;
        for (int i = 0 ; i < N ; i++){
            particle[i] = parent[i][idx];
        }
        return particle ;
    }

    vector<G> daughterEstimate(){
        int maxIdx = 0 ;
        int maxWeight = -1 ;
        for (int i = 0 ; i < nParticles ; i++){
            if (parentWeights[i] > maxWeight){
                maxIdx = i ;
                maxWeight = parentWeights[i] ;
            }
        }
        vector<G> estimate ;
        vector<G> maxDaughter = daughter[maxIdx] ;
        DEBUG_VAL(maxDaughter.size()) ;
        remove_copy_if(maxDaughter.begin(),maxDaughter.end(),
                       estimate.begin(),weightBelow<G>) ;
        return estimate ;
    }

private:
    // device pointers
    double* dParent ;
    double* dMeasurements ;
    G* dDaughter ;
    int* dOffsets ;


    int verbosity_ ;
    vector<int> offsets ;
    vector<G> concat ;

    void initializeParent(int nParticles, vector<double> initValue){
        parent.clear();
        parent.resize(N);
        for (int n = 0 ; n < N ; n++ ){
            parent[n].clear() ;
            parent[n].assign(nParticles,initValue[n]) ;
        }
        parentWeights.clear();
        parentWeights.assign(nParticles, 1.0/nParticles);

        // allocate device memory for particles
        cudaMalloc((void**) &dParent, nParticles*N*sizeof(double)) ;

        copyParent();
    }

    void initializeParent(int n){
        vector<double> initValue(N,0.0) ;
        initializeParent(n,initValue) ;
    }

    void initializeDaughter(int nParticles){
        daughter.resize(nParticles);
    }

    void computeOffsets (){
        vector<int> offsets(nParticles+1,0) ;
        int sum = 0 ;
        for (int i = 0; i < nParticles ; i++){
            offsets[i] = sum ;
            sum += daughter[i].size() ;
        }
        offsets[nParticles] = sum ;

        size_t offsetSize = sizeof(int)*(nParticles+1) ;
        cudaFree(dOffsets) ;
        cudaMalloc((void**)&dOffsets,offsetSize) ;
        cudaMemcpy(dOffsets,offsets.data(),offsetSize,cudaMemcpyHostToDevice) ;
    }

    void copyParent(){
        vector<double> flattened(N*nParticles) ;
        for (int i = 0 ; i < N ; i++){
            flattened.insert(flattened.end(),parent[i].begin(),parent[i].end()) ;
        }
        cudaMemcpy(dParent,flattened.data(),
                   N*nParticles*sizeof(double),cudaMemcpyHostToDevice) ;
    }

    void copyDaughter(){
        concat.clear();
        for (int i = 0 ; i < daughter.size() ; i++){
            concat.insert(concat.end(),daughter[i].begin(),daughter[i].end()) ;
        }
        cudaFree(dDaughter) ;
        nDaughter = concat.size() ;
        size_t daughterSize = concat.size() * sizeof(G) ;
        cudaMalloc( (void**)&dDaughter, daughterSize) ;
        cudaMemcpy(dDaughter,concat.data(),daughterSize,cudaMemcpyHostToDevice) ;
    }

    void copyMeasurements(vector<vector<double>> measurements){
        vector<double> measurementsFlattened ;
        for (int i = 0 ; i < M ; i++){
            measurementsFlattened.insert(measurementsFlattened.end(),
                                         measurements[i].begin(),
                                         measurements[i].end()) ;
        }
        nMeasure = measurements[0].size() ;
        size_t measureSize = M*nMeasure*sizeof(double) ;
        cudaFree(dMeasurements) ;
        cudaMalloc((void**)&dMeasurements,measureSize) ;
        cudaMemcpy(dMeasurements,measurementsFlattened.data(),measureSize,
                   cudaMemcpyHostToDevice) ;
    }

    void readDaughterUpdate(G* dUpdate){
        size_t updateSize = (nDaughter*(nMeasure+1) + nParticles*nMeasure) ;
        concat.resize(updateSize);
        cudaMemcpy(concat.data(),dUpdate,updateSize*sizeof(G),cudaMemcpyDeviceToHost) ;
        typename vector< G >::iterator start ;
        typename vector< G >::iterator stop ;
        for ( int i = 0 ; i < nParticles ; i++){
            start = concat.begin() + (nMeasure+1)*offsets[i] + nMeasure*i ;
            stop = concat.begin() + (nMeasure+1)*offsets[i+1] + nMeasure*(i+1) ;
            daughter[i].assign(start,stop) ;
        }
    }

    void predictParent(){
        for (int i = 0 ; i < nParticles ; i++ ){

            vector<double> priorParticle = getParticle(i) ;
            vector<double> predictedParticle = parentMotionModel(priorParticle) ;
            for (int n = 0 ; n < N ; n++){
                parent[n][i] = predictedParticle[n] ;
            }
        }

//        for (int i = 0 ; i < nParticles; i++){
//            vector<double> p = getParticle(i) ;
//            cout << "[" << p[0] << "," << p[1] << "]" << endl ;
//        }
    }
};

#endif // SCPHD_CUH
