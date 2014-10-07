#ifndef SCPHD_CUH
#define SCPHD_CUH

#include <vector>
#include <ctime>
#include <thread>
#include <future>
#include <nvToolsExtCuda.h>

#include "kernels.cuh"
#include "types.h"
#include "gmreduction.h"
#include "math.h"

#define DEBUG_MSG(x) if ( verbosity_ >= 2) cout << "[" << __func__ << "(" << __LINE__ << ")]: " << x << endl
#define DEBUG_VAL(x) if ( verbosity_ >= 2) cout << "[" << __func__ << "(" << __LINE__ << ")]: " << #x << " = " << x << endl

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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
    vector< vector< vector<double> > > distances ;
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
    size_t maxGlobalMem ;

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

        nDaughter = 0 ;
        nMeasure = 0 ;
        dOffsets = NULL ;
        dDaughter = NULL ;
        dParent = NULL ;
        dMeasurements = NULL ;

        initializeGPU() ;
        initializeParent(nParticles);
        initializeDaughter(nParticles) ;
        offsets.assign(nParticles+1,0);
        distances.resize(nParticles);
    }


    ~SCPHDFilter(){
        gpuErrchk(cudaFree((void*)dParent)) ;
        gpuErrchk(cudaFree((void*)dDaughter)) ;
        gpuErrchk(cudaFree((void*)dMeasurements)) ;
    }

    void predict(){
//        cout << "predictParent" << endl ;
         predictParent() ;

//        cout << "computeOffsets" << endl ;
        computeOffsets() ;

        cout << "copyDaughter" << endl ;
        DEBUG_VAL(nDaughter) ;
        copyDaughter() ;
        DEBUG_VAL(nDaughter) ;

        int gridDim = nParticles ;
        int blockDim = 256 ;
//        cout << "daughterPredictKernel" << endl ;
        daughterPredictKernel<G,DaughterMM,N><<<gridDim,blockDim>>>(dDaughter,dParent,
                                                  daughterMotionModel,
                                                  dOffsets, nParticles) ;
        gpuErrchk(cudaPeekAtLastError()) ;
//        gpuErrchk(cudaDeviceSynchronize()) ;
        readDaughterPredict(dDaughter);
        copyParent();
    }

    void update(vector< vector<double> > measurements){
        copyMeasurements(measurements);
        DEBUG_VAL(nMeasure) ;
        dim3 gridDim ;
        dim3 blockDim ;

        vector<size_t> requiredMemory = computeRequiredMemory() ;
        int startIdx = 0 ;
        int endIdx = 0 ;

        size_t memfree ;
        size_t total ;
        size_t batchBytes ;

        while (endIdx < nParticles){
            gpuErrchk(cudaMemGetInfo(&memfree,&total)) ;
            batchBytes = 0 ;
            startIdx = endIdx ;
            while(batchBytes < 0.9*memfree){
                batchBytes += requiredMemory[endIdx++] ;
                if (endIdx == nParticles)
                    break ;
            }

            DEBUG_VAL(startIdx) ;
            DEBUG_VAL(endIdx) ;            

            DEBUG_VAL(batchBytes) ;
            DEBUG_VAL(memfree) ;

            computeOffsets(startIdx,endIdx);
            copyParent(startIdx,endIdx);
            copyDaughter(startIdx,endIdx);
            int batchParticles = endIdx - startIdx ;

            DEBUG_VAL(batchParticles) ;
            DEBUG_VAL(nDaughter) ;
            DEBUG_VAL(nMeasure) ;

//            for (int i = startIdx ; i < endIdx ; i++){
//                cout << "offset["<<i<<"] = " << offsets[i] << endl ;
//            }

            G* dUpdate ;
            size_t updateSize = sizeof(G)*(batchParticles*nMeasure +
                                           nMeasure*nDaughter +
                                           nDaughter) ;
            gpuErrchk(cudaMalloc((void**)&dUpdate,updateSize)) ;

            // compute births
            if (nMeasure > 0){
                DEBUG_MSG("births") ;
                gridDim.x = batchParticles ;
                gridDim.y = 1 ;
                blockDim.x = 256 ;
                blockDim.y = 1;

                computeBirthsKernel<G,MeasureModel,N,M><<<gridDim,blockDim>>>
                    (dMeasurements,dParent,measureModel,w0,
                     dOffsets,nMeasure,dUpdate) ;
                gpuErrchk(cudaPeekAtLastError()) ;                
//                gpuErrchk(cudaDeviceSynchronize()) ;
            }


            // compute detections
            if (nDaughter > 0){
                DEBUG_MSG("detections") ;
                gridDim.x = batchParticles ;
                blockDim.x = 32 ;
                blockDim.y = 32 ;
                computeDetectionsKernel<G,MeasureModel,N,M><<<gridDim,blockDim>>>
                    (dParent,dDaughter,dMeasurements, measureModel,
                     dOffsets,nMeasure,dUpdate) ;
                gpuErrchk(cudaPeekAtLastError()) ;
//                gpuErrchk(cudaDeviceSynchronize()) ;
            }

            // compute weight normalizers
            double* dNormalizers = NULL ;
            size_t normalizerSize = nMeasure*batchParticles*sizeof(double) ;
            gpuErrchk(cudaMalloc((void**)&dNormalizers,normalizerSize)) ;
            if (nDaughter > 0){
                DEBUG_MSG("normalizers") ;
                gridDim.x = min(batchParticles,1024) ;
                gridDim.y = min(nMeasure,32) ;
                blockDim.x = 256 ;
                blockDim.y = 1 ;
                computeNormalizersKernel<G><<<gridDim,blockDim>>>
                    (dUpdate,w0,kappa,dOffsets,batchParticles,
                     nMeasure,dNormalizers) ;
                gpuErrchk(cudaPeekAtLastError()) ;
//                gpuErrchk(cudaDeviceSynchronize()) ;

            }
            else{
                DEBUG_MSG("compute normalizers without detection terms") ;
                vector<double> hNormalizers(nMeasure*batchParticles,w0+kappa) ;
                gpuErrchk(cudaMemcpy(dNormalizers,&hNormalizers[0],normalizerSize,
                        cudaMemcpyHostToDevice)) ;
            }

            // re-weight parent particles
            DEBUG_MSG("weight parents") ;
            vector<double> normalizers(nMeasure*batchParticles,1.0) ;
            gpuErrchk(cudaMemcpy(&normalizers[0],dNormalizers,
                                 normalizerSize,cudaMemcpyDeviceToHost)) ;

            for (int i = startIdx ; i < endIdx ; i++){
                for (int m = 0 ; m < nMeasure ; m++){
                    int idx = (i-startIdx)*nMeasure + m ;
                    parentWeights[i] *= normalizers[idx] ;
                }
                double cnPredict = 0 ;
                for (int j = 0 ; j < daughter[i].size() ; j++){
                    cnPredict += daughter[i][j].weight ;
                }
                parentWeights[i] *= exp(-cnPredict) ;
            }

            // do the update
            DEBUG_MSG("complete update") ;
            gridDim.x = batchParticles ;
            gridDim.y = 1 ;
            blockDim.x = 256 ;
            blockDim.y = 1 ;
            updateKernel<G><<<gridDim,blockDim>>>
                (dDaughter,dNormalizers, pd,
                 dOffsets, nMeasure, dUpdate) ;
            gpuErrchk(cudaPeekAtLastError()) ;
//            gpuErrchk(cudaDeviceSynchronize()) ;

//            // compute distances for GM reduction
//            int nUpdateSquared = 0 ;
//            for (int n = startIdx ; n < endIdx ; n++){
//                int m = daughter[n].size()*(nMeasure+1) + nMeasure ;
//                nUpdateSquared += m*m ;
//                distances[n].resize(m) ;
//                for ( int k = 0 ; k < m ; k++){
//                    distances[n][k].assign(m,0.0) ;
//                }
//            }
//            double* dDistances = NULL ;
//            gpuErrchk(cudaMalloc((void**)&dDistances,
//                                 nUpdateSquared*sizeof(double))) ;


//            gridDim.x = batchParticles ;
//            gridDim.y = 1 ;
//            blockDim.x = 32 ;
//            blockDim.y = 32 ;
//            computeDistances<G><<<gridDim,blockDim>>>(dUpdate,dOffsets,
//                                                      nMeasure,dDistances) ;
//            gpuErrchk(cudaPeekAtLastError()) ;
//            gpuErrchk(cudaDeviceSynchronize()) ;

//            double* ptr = dDistances ;
//            for (int n = startIdx ; n < endIdx ; n++){
//                int m = distances[n].size() ;
//                for (int k = 0 ; k < m ; k++){
//                    gpuErrchk(cudaMemcpy(distances[n][k].data(),ptr,
//                                         m*sizeof(double),cudaMemcpyDeviceToHost)) ;
//                    ptr += m ;
//                }
//            }


            DEBUG_MSG("copy daughter to host") ;
            readDaughterUpdate(dUpdate,startIdx,endIdx);


            // free memory
            DEBUG_MSG("cudaFree") ;
            gpuErrchk(cudaFree(dUpdate)) ;
            gpuErrchk(cudaFree(dNormalizers)) ;
//            gpuErrchk(cudaFree(dDistances)) ;
        }

        // GM reduction
        DEBUG_MSG("reduce gaussian mixture") ;
        int concurrency = thread::hardware_concurrency() ;
        DEBUG_VAL(concurrency) ;
        nvtxRangePushA("GM reduction") ;
        if (concurrency > 2){
            concurrency -= 1 ;
            int n = 0 ;
            vector< future< vector <G> > > futures ;
            vector<G> (*fn)(vector<G>, double,double) ;
            fn = &reduceGaussianMixture<G> ;
            while ( n < nParticles){
//                DEBUG_VAL(n) ;
                // start threads
                futures.clear();
//                DEBUG_MSG("async") ;
                for ( int i = 0 ; i < concurrency ; i++){
                    if ( n < nParticles ){
//                        DEBUG_VAL(distances[n].size()) ;
//                        DEBUG_VAL(distances[n][0].size()) ;
                        futures.push_back(async(launch::async,fn,daughter[n++],
//                                                distances[n],
                                                pruneWeight, mergeDist));
                    }
                    else{
                        break ;
                    }
                }

//                DEBUG_MSG("get") ;
                // synchronize threads
                n -= futures.size() ;
                for ( int i = 0 ; i < concurrency ; i++){
                    if (n < nParticles){
                        daughter[n++] = futures[i].get() ;
                    }
                }
            }
        }
        else{
            for (int n = 0 ; n < nParticles ; n++){
                daughter[n] = reduceGaussianMixture<G>(daughter[n],
//                                                       distances[n],
                                                       pruneWeight,
                                                       mergeDist) ;
            }
        }
        nvtxRangePop() ;

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
        {
            cout << "not resampling" << endl ;
            return ;
        }

        nvtxRangeId_t rId = nvtxRangeStartA("Resample parent") ;
        cout << "resampling" << endl ;
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
            resampledDaughter[j].assign(daughter[i].begin(),daughter[i].end()) ;
        }
        parent = resampledParent ;
        daughter = resampledDaughter ;
        parentWeights.assign(nParticles,interval);
        nvtxRangeEnd(rId);
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
        vector<G> maxDaughter = daughter[maxIdx] ;
        DEBUG_VAL(maxDaughter.size()) ;

        vector<G> estimate(maxDaughter.size()) ;
        typename vector<G>::iterator estimateEnd =
                remove_copy_if(maxDaughter.begin(),maxDaughter.end(),
                               estimate.begin(),weightBelow<G>) ;
        estimate.resize(estimateEnd - estimate.begin());
        DEBUG_VAL(estimate.size()) ;
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

    void initializeGPU(){
        int nDevices ;
        cudaGetDeviceCount( &nDevices ) ;
        cout << "Found " << nDevices << " CUDA Devices" << endl ;
        if (nDevices == 0){
            cout << "No compatible devices found. Exiting." << endl ;
            exit(-1) ;
        }
        cudaDeviceProp props ;
        gpuErrchk(cudaGetDeviceProperties( &props, 0 )) ;
        cout << "Device name: " << props.name << endl ;
        cout << "Compute capability: " << props.major << "." << props.minor << endl ;
        cout << "Total global memory: " << props.totalGlobalMem << endl ;
        maxGlobalMem = props.totalGlobalMem ;
    }

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
        gpuErrchk(cudaMalloc((void**) &dParent, nParticles*N*sizeof(double))) ;

        copyParent();
    }

    void initializeParent(int n){
        vector<double> initValue(N,0.0) ;
        initializeParent(n,initValue) ;
    }

    void initializeDaughter(int nParticles){
        daughter.resize(nParticles);
    }

    void computeOffsets(){
        computeOffsets(0,nParticles);
    }

    void computeOffsets (int start, int end){
        offsets.resize(end-start+1,0) ;
        int sum = 0 ;
        for (int i = start; i < end ; i++){
            offsets[i] = sum ;
            sum += daughter[i].size() ;
        }
        offsets[end] = sum ;

        size_t offsetSize = sizeof(int)*(end-start+1) ;
        if (dOffsets != NULL)
            gpuErrchk(cudaFree(dOffsets)) ;
        gpuErrchk(cudaMalloc((void**)&dOffsets,offsetSize)) ;
        gpuErrchk(cudaMemcpy(dOffsets,offsets.data(),offsetSize,cudaMemcpyHostToDevice)) ;
    }

    void copyParent(){
        copyParent(0,nParticles);
    }

    void copyParent(int start,int end){
        int n = end - start ;
        vector<double> flattened(N*n) ;
        for (int i = 0 ; i < N ; i++){
            flattened.insert(flattened.end(),
                             parent[i].begin()+start,
                             parent[i].begin()+end) ;
        }
        if (dParent != NULL)
            gpuErrchk(cudaFree(dParent)) ;
        gpuErrchk(cudaMalloc((void**)&dParent,N*n*sizeof(double)))
        gpuErrchk(cudaMemcpy(dParent,flattened.data(),
                   N*n*sizeof(double),cudaMemcpyHostToDevice)) ;
    }

    void copyDaughter(){
        copyDaughter(0,nParticles);
    }

    void copyDaughter(int startIdx, int endIdx){
        nvtxRangeId_t rId = nvtxRangeStartA("copyDaughter") ;
        concat.clear();
        for (int i = startIdx ; i < endIdx ; i++){
//            DEBUG_VAL(daughter[i].size()) ;
            concat.insert(concat.end(),daughter[i].begin(),daughter[i].end()) ;
        }
        if (dDaughter != NULL)
            gpuErrchk(cudaFree(dDaughter)) ;
        nDaughter = concat.size() ;
        size_t daughterSize = nDaughter * sizeof(G) ;
        gpuErrchk(cudaMalloc( (void**)&dDaughter, daughterSize)) ;
        gpuErrchk(cudaMemcpy(dDaughter,concat.data(),daughterSize,
                             cudaMemcpyHostToDevice)) ;
        nvtxRangeEnd(rId);
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
        gpuErrchk(cudaFree(dMeasurements)) ;
        gpuErrchk(cudaMalloc((void**)&dMeasurements,measureSize)) ;
        gpuErrchk(cudaMemcpy(dMeasurements,measurementsFlattened.data(),
                             measureSize,cudaMemcpyHostToDevice)) ;
    }

    vector<size_t> computeRequiredMemory(){
        vector<size_t> requiredMemory(nParticles) ;
//        size_t sum = nDaughter*sizeof(G) ; // predicted daughter
//        sum += N*nParticles*sizeof(double) ; // parent particles
//        sum += M*sizeof(double) ; // measurements
        size_t sum = 0 ;
        for ( int i = 0 ; i < nParticles ; i++){
            int nFeatures = daughter[i].size() ;
            int nUpdate = nFeatures*(nMeasure+1) + nMeasure ;
            sum = nUpdate*sizeof(G)
                + (N // parent particle
//                + nUpdate*nUpdate // merge distances
                + nMeasure) // normalizers
                *sizeof(double)
                + sizeof(int) ;// offset
            requiredMemory[i] = sum ;
        }
        return requiredMemory ;
    }

    void readDaughterPredict(G* ptr){
        readDaughterPredict(ptr,0,nParticles);
    }

    void readDaughterPredict(G* ptr, int startIdx,int endIdx){
        nvtxRangeId_t rId = nvtxRangeStartA("readDaughterPredict") ;
        for ( int i = startIdx ; i < endIdx ; i++){
            int n = daughter[i].size() ;
            daughter[i].resize(n) ;
//            DEBUG_VAL(i) ;
            gpuErrchk(cudaMemcpy(daughter[i].data(),ptr,
                                 n*sizeof(G),cudaMemcpyDeviceToHost)) ;
            ptr += n ;
        }
        nvtxRangeEnd(rId);
    }

    void readDaughterUpdate(G* ptr){
        readDaughterUpdate(ptr,0,nParticles);
    }

    void readDaughterUpdate(G* ptr, int startIdx,int endIdx){
        nvtxRangeId_t rId = nvtxRangeStartA("readDaughterUpdate") ;
        int nPredict = 0 ;
        for (int i = startIdx ; i < endIdx ; i++){
            nPredict += daughter[i].size() ;
        }
        int nUpdate = nPredict*(nMeasure+1) + (endIdx-startIdx)*nMeasure ;
        concat.resize(nUpdate);
        gpuErrchk(cudaMemcpy(concat.data(),ptr,
                             nUpdate*sizeof(G),
                             cudaMemcpyDeviceToHost)) ;

        typename vector<G>::iterator it = concat.begin() ;
        for ( int i = startIdx ; i < endIdx ; i++){
            int n = daughter[i].size()*(nMeasure+1) + nMeasure ;
            daughter[i].resize(n) ;
            daughter[i].assign(it,it+n) ;
            it += n ;
//            DEBUG_VAL(i) ;
//            gpuErrchk(cudaMemcpy(daughter[i].data(),ptr,
//                                 n*sizeof(G),cudaMemcpyDeviceToHost)) ;
//            ptr += n ;
        }
        nvtxRangeEnd(rId);
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
