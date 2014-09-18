#ifndef MODELS_CUH
#define MODELS_CUH

#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "math.h"

using namespace std ;

template <int N>
class BrownianMotionParentModel{
public:
    double sigma[N] ;

    BrownianMotionParentModel(){
        for (int i = 0 ; i < N ; i++){
            sigma[i] = 1.0 ;
        }
    }

    BrownianMotionParentModel(double s[]){
        for (int i = 0 ; i < N ; i++){
            sigma[i] = s[i] ;
        }
    }

    vector<double> operator() (vector<double> prior){
        vector<double> predicted ;
        for (int i = 0 ; i < N ;i++){
            double displacement = randn<double>()*sigma[i] ;
            predicted.push_back(prior[i] + displacement) ;
        }
        return predicted ;
    }
};

template <int N, class G>
class BrownianMotionDaughterModel{
public:
    double sigma[N] ;

    BrownianMotionDaughterModel(){
        for (int i = 0 ; i < N ; i++){
            sigma[i] = 0.0 ;
        }
    }

    BrownianMotionDaughterModel(double s[]){
        for (int i = 0 ; i < N ; i++){
            sigma[i] = s[i] ;
        }
    }

    __device__ G operator() (G daughter, double* parent){
        G predicted = daughter ;
        for (int i = 0 ; i < N ; i++ ){
            predicted.cov[i*i] += sigma[i] ;
        }
        return predicted ;
    }
};

template <int N,int N2 = N*N>
class BiasedIdentityModel{
public:
    double R[N2] ;
    double pd ;

    BiasedIdentityModel(){
        for (int i = 0 ; i < N2 ; i++){
            R[i] = 0.0 ;
        }
        pd = 1.0 ;
    }

    BiasedIdentityModel(double* r, double prob_detect){
        for (int i = 0 ; i < N2 ; i++){
            R[i] = r[i] ;
        }
        pd = prob_detect ;
    }

    /// compute detection
    __device__ __host__ Gaussian<N,N2>
    operator() (double* parent, Gaussian<N,N2>daughter, double* measurement){
        Gaussian<N,N2> g ;
        Gaussian<N,N2> updated ;
        double predicted_measurement[N] ;
        double likelihood ;
        g.weight = 1.0 ;
        for (int i = 0 ; i < N2 ; i++){
            if ( i < N ){
                g.mean[i] = measurement[i] ;
                predicted_measurement[i] = daughter.mean[i] + parent[i] ;
            }
            g.cov[i] = R[i] ;
        }
        likelihood = evalGaussian(g,predicted_measurement) ;

        double H[N2] ;
        fill_identity_matrix(H,N);

        double innovation[N] ;
        for (int i = 0 ; i < N ; i++){
            innovation[i] = measurement[i] - predicted_measurement[i] ;
        }
        updated = ekfUpdate<N,N>(daughter,innovation,H,R) ;
        updated.weight = daughter.weight*likelihood ;
        return updated ;
    }

    /// compute birth
    __device__ __host__ Gaussian<N,N2>
    operator() (double* parent, double* measurement,double w0){
        Gaussian<N,N2> birth ;

        birth.weight = w0 ;
        for (int i = 0 ; i < N2 ; i++){
            if ( i < N ){
                birth.mean[i] = measurement[i] - parent[i] ;
            }
            birth.cov[i] = R[i] ;
        }
        return birth ;
    }
};

#endif // MODELS_CUH
