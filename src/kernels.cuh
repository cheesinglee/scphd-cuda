#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include "math.h"
#include "types.h"

/// predict daughter gaussians, in-place modification
template <class G, class Functor,int N>
__global__ void
daughterPredictKernel(G* daughterArray, double* parentArray,
                      Functor motionModel, int* offsets, int nParticles){

    int tid = threadIdx.x ;
    int idxParent = blockIdx.x ;
    int offset = offsets[idxParent] ;
    int nFeatures = offsets[idxParent+1] - offset ;
    double priorParent[N] ;
    for (int i = 0 ; i < N ; i++){
        priorParent[i] = parentArray[idxParent + i*nParticles] ;
    }
    for (int j = tid ; j < nFeatures ; j += blockDim.x ){
        int idx = offset + j ;
        G priorFeature = daughterArray[idx] ;
        daughterArray[idx] = motionModel(priorFeature,priorParent) ;
    }
}

template <class G, class Functor,int N,int M>
__global__ void
computeBirthsKernel(double* measurements, double* parent,
                    Functor measurementModel,
                    double w0, int nMeasure,int nParticles,
                    G* birth){
    int tid = threadIdx.x ;
    int parentIdx = blockIdx.x ;
    double particle[N] ;
    for (int i = 0 ;i < N ; i++){
        particle[i] = parent[parentIdx + i*nParticles] ;
    }

    for (int j = tid ; j < nMeasure ; j += blockDim.x){
        // get the measurement
        double z[M] ;
        for (int m = 0 ; m < M ; m++){
            z[m] = measurements[j + m*nMeasure] ;
        }

        int birthIdx = parentIdx*nMeasure + j ;
//        printf("measurement = [%f,%f]  parent = [%f,%f]\n",z[0],z[1],particle[0],particle[1]) ;
        birth[birthIdx] = measurementModel(particle,z,w0) ;
    }
}

template <class G,class Functor, int N, int M>
__global__ void
computeDetectionsKernel(double* parent, G* daughter, double* measurements,
                        Functor measurementModel, int* offsets, int nMeasure,
                        int nPredict, G* detections){
    int tid = threadIdx.x ;
    int zid = threadIdx.y ;
    int parentIdx = blockIdx.x ;
    int offset = offsets[parentIdx] ;
    int nFeatures = offsets[parentIdx+1] - offset ;
    double parentParticle[N] ;
    for (int i = 0 ; i < N ; i++){
        parentParticle[i] = parent[parentIdx + i*gridDim.x] ;
    }

    for (int j = tid ; j < nFeatures ; j+=blockDim.x){
        for (int m = zid ; m < nMeasure ; j+=blockDim.y){
            double z[M] ;
            for (int i = 0 ; i < M ; i++){
                z[i] = measurements[m+i*nMeasure] ;
            }
            G predicted = daughter[j+offset] ;
            G updated = measurementModel(parentParticle,predicted,z) ;
            detections[offset+j+nPredict*m] = updated ;
        }
    }
}

template <class G>
__global__ void
updateKernel(G* predicted, G* detections,G* births, double* normalizers,
             double pd, int* offsets,int nMeasure, G* updated){
    int tid = threadIdx.x ;
    int parentIdx = blockIdx.x ;
    int predictOffset = offsets[parentIdx] ;
    int updateOffset = predictOffset*nMeasure + predictOffset + parentIdx*nMeasure ;
    int nDaughter = offsets[parentIdx+1] - predictOffset ;
    int nUpdate = nDaughter*nMeasure + nDaughter + nMeasure ;
    for (int j = tid ;j < nUpdate ; j+=blockDim.x){
        int updateIdx = updateOffset + j ;
        G feature ;
        int detectBegin = nDaughter ;
        int birthBegin = nDaughter*(nMeasure+1) ;
        if ( j < detectBegin ){
            // non-detect term
            feature = predicted[j+predictOffset] ;
            feature.weight *= (1-pd) ;
        }
        else if ( j >= detectBegin && j < birthBegin ){
            // detection term
            int measureIdx = floor(double(j)/(double)nDaughter) ;
            int featureIdx = fmod(double(j),(double)nDaughter) ;
            int detectIdx = nMeasure*predictOffset
                    + measureIdx*nDaughter + featureIdx ;
            feature = detections[detectIdx] ;

            int idx_normalizer = measureIdx + parentIdx*nMeasure ;
            feature.weight /= normalizers[idx_normalizer] ;
//            feature.weight = exp(feature.weight) ;
        }
        else if (j >= nDaughter*nMeasure ){
            // birth term
            int idx = j - (nDaughter*nMeasure) ;
            feature = births[idx] ;

            int measureIdx = idx ;
            int idx_normalizer = measureIdx + parentIdx*nMeasure ;
            feature.weight /= normalizers[idx_normalizer] ;
//            feature.weight = exp(feature.weight) ;
        }
        updated[updateIdx] = feature ;
    }
}

template <class G>
__global__ void
computeNormalizersKernel(G* detections, double w0, double kappa,
                         int* offsets, int nParticles, int nMeasure,
                         double* normalizers){
    __shared__ double sdata[256] ;
    int tid = threadIdx.x ;
    double sum ;

    for (int parentIdx = blockIdx.x ; parentIdx < nParticles ; parentIdx+=gridDim.x){
        for(int measureIdx = blockIdx.y ; measureIdx < nMeasure ; measureIdx += gridDim.y){
            int nDaughter = offsets[parentIdx+1] - offsets[parentIdx] ;
            int offset = offsets[parentIdx]*nMeasure + nDaughter*measureIdx ;
            sum = 0.0 ;
            for (int i = tid ; i < nDaughter ; i += blockDim.x){
                sum += detections[offset+i].weight ;
            }
            sumByReduction(sdata,sum,tid);
            __syncthreads() ;
            if (tid == 0){
                int idx = nMeasure*parentIdx + measureIdx ;
                normalizers[idx] = sdata[0] + w0 + kappa ;
            }
            __syncthreads() ;
        }
    }

    tid = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*gridDim.x*blockDim.x ;
    if (tid < nParticles*nMeasure)
        printf("normalizers[%d] = %f\n",tid,normalizers[tid]) ;
}


#endif // KERNELS_CUH
