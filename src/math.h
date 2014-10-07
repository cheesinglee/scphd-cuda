#ifndef MATH_H
#define MATH_H

#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cuda_runtime.h>
#include "types.h"

using namespace std ;

#define TWO_PI 6.2831853071795864769252866

/// uniform random number in range [0,1]
template <class T>
T randu(){
    T randval = rand() /((T) RAND_MAX) ;
    return randval ;
}

/// normally distributed random number, mean 0, variance 1.
template <class T>
T randn(){
    static bool haveSpare = false;
    static T rand1, rand2;

    if(haveSpare)
    {
        haveSpare = false;
        return sqrt(rand1) * sin(rand2);
    }

    haveSpare = true;

    rand1 = rand() / ((T) RAND_MAX);
    if(rand1 < 1e-100) rand1 = 1e-100;
    rand1 = -2 * log(rand1);
    rand2 = (rand() / ((T) RAND_MAX)) * TWO_PI;

    return sqrt(rand1) * cos(rand2);
}

//*********** Matrix functions *********************

__device__ __host__ void
fill_identity_matrix(double* A, int dims){
    for ( int i = 0; i < dims ;i++){
        for(int j =0; j < dims ;j++){
            A[i+j*dims] = (i==j) ;
        }
    }
}

/// matrix multiplication X = A*B
/// matrix elements stored in column-major order
template <int ROWS_A, int COLS_A, int COLS_B>
__host__ __device__
void matmultiply(double* A, double* B, double* X){
    for (int i = 0 ; i < ROWS_A ; i++){
        for(int j = 0 ; j < COLS_B ; j++){
            double val = 0 ;
            for (int k = 0 ; k < COLS_A ; k++){
                val += A[i + k*ROWS_A]*B[k + j*COLS_A] ;
            }
            X[i+j*ROWS_A] = val ;
        }
    }
}

/// specialization for multiplying two 2x2 matrices
template <>
__host__ __device__
void matmultiply<2,2,2>(double* A, double* B, double* X){
    X[0] = A[0]*B[0] + A[2]*B[1] ;
    X[1] = A[1]*B[0] + A[3]*B[1] ;
    X[2] = A[0]*B[2] + A[2]*B[3] ;
    X[3] = A[1]*B[2] + A[3]*B[3] ;
}

/// transpose a matrix
template <int M, int N>
__host__ __device__
void transpose(double* A, double* At){
    for (int i = 0 ; i < M ; i++){
        for (int j = 0 ; j < N ; j++){
            At[j + i*N] = A[i + j*M] ;
        }
    }
}

/// determinant of a 2x2 matrix
__host__ __device__ double
det2(double *A){
    return A[0]*A[3] - A[2]*A[1] ;
}

/// determinant of a 3x3 matrix
__host__ __device__ double
det3(double *A){
    return (A[0]*A[4]*A[8] + A[3]*A[7]*A[2] + A[6]*A[1]*A[5])
        - (A[0]*A[7]*A[5] + A[3]*A[1]*A[8] + A[6]*A[4]*A[2]) ;
}

/// determinant of a 4x4 matrix
__host__ __device__ double
det4(double *A)
{
    double det=0;
    det+=A[0]*((A[5]*A[10]*A[15]+A[9]*A[14]*A[7]+A[13]*A[6]*A[11])-(A[5]*A[14]*A[11]-A[9]*A[6]*A[15]-A[13]*A[10]*A[7]));
    det+=A[4]*((A[1]*A[14]*A[11]+A[9]*A[2]*A[15]+A[13]*A[10]*A[3])-(A[1]*A[10]*A[15]-A[9]*A[14]*A[3]-A[13]*A[2]*A[11]));
    det+=A[8]*((A[1]*A[6]*A[15]+A[5]*A[14]*A[3]+A[13]*A[2]*A[7])-(A[1]*A[14]*A[7]-A[5]*A[2]*A[15]-A[13]*A[6]*A[3]));
    det+=A[12]*((A[1]*A[10]*A[7]+A[5]*A[2]*A[12]+A[9]*A[10]*A[3])-(A[1]*A[10]*A[12]-A[5]*A[10]*A[3]-A[9]*A[2]*A[7]));
    return det ;
}

template <int N>
__device__ __host__ void
invert_matrix(double *A, double *AInv){}

/// invert a 2x2 matrix
template <>
__device__ __host__ void
invert_matrix<2>(double *A, double *A_inv)
{
    double det = det2(A) ;
    A_inv[0] = A[3]/det ;
    A_inv[1] = -A[1]/det ;
    A_inv[2] = -A[2]/det ;
    A_inv[3] = A[0]/det ;
}

/// invert a 3x3 matrix
template <>
__device__ __host__ void
invert_matrix<3>(double *A, double* A_inv){
    double det = det3(A) ;
    A_inv[0] = (A[4]*A[8] - A[7]*A[5])/det ;
    A_inv[1] = (A[7]*A[2] - A[1]*A[8])/det ;
    A_inv[2] = (A[1]*A[5] - A[4]*A[2])/det ;
    A_inv[3] = (A[6]*A[5] - A[3]*A[8])/det ;
    A_inv[4] = (A[0]*A[8] - A[6]*A[2])/det ;
    A_inv[5] = (A[2]*A[3] - A[0]*A[5])/det ;
    A_inv[6] = (A[3]*A[7] - A[6]*A[4])/det ;
    A_inv[7] = (A[6]*A[1] - A[0]*A[7])/det ;
    A_inv[8] = (A[0]*A[4] - A[3]*A[1])/det ;
}

template<>
/// invert a 4x4 matrix
__device__ __host__ void
invert_matrix<4>( double *A, double *Ainv)
{
    Ainv[0] = (A[5] * A[15] * A[10] - A[5] * A[11] * A[14] - A[7] * A[13] * A[10] + A[11] * A[6] * A[13] - A[15] * A[6] * A[9] + A[7] * A[9] * A[14]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[1] = -(A[15] * A[10] * A[1] - A[11] * A[14] * A[1] + A[3] * A[9] * A[14] - A[15] * A[2] * A[9] - A[3] * A[13] * A[10] + A[11] * A[2] * A[13]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[2] = (A[5] * A[3] * A[14] - A[5] * A[15] * A[2] + A[15] * A[6] * A[1] + A[7] * A[13] * A[2] - A[3] * A[6] * A[13] - A[7] * A[1] * A[14]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[3] = -(A[5] * A[3] * A[10] - A[5] * A[11] * A[2] - A[3] * A[6] * A[9] - A[7] * A[1] * A[10] + A[11] * A[6] * A[1] + A[7] * A[9] * A[2]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[4] = -(A[15] * A[10] * A[4] - A[15] * A[6] * A[8] - A[7] * A[12] * A[10] - A[11] * A[14] * A[4] + A[11] * A[6] * A[12] + A[7] * A[8] * A[14]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[5] = (A[0] * A[15] * A[10] - A[0] * A[11] * A[14] + A[3] * A[8] * A[14] - A[15] * A[2] * A[8] + A[11] * A[2] * A[12] - A[3] * A[12] * A[10]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[6] = -(A[0] * A[15] * A[6] - A[0] * A[7] * A[14] - A[15] * A[2] * A[4] - A[3] * A[12] * A[6] + A[3] * A[4] * A[14] + A[7] * A[2] * A[12]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[7] = (-A[0] * A[7] * A[10] + A[0] * A[11] * A[6] + A[7] * A[2] * A[8] + A[3] * A[4] * A[10] - A[11] * A[2] * A[4] - A[3] * A[8] * A[6]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[8] = (-A[5] * A[15] * A[8] + A[5] * A[11] * A[12] + A[15] * A[4] * A[9] + A[7] * A[13] * A[8] - A[11] * A[4] * A[13] - A[7] * A[9] * A[12]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[9] = -(A[0] * A[15] * A[9] - A[0] * A[11] * A[13] - A[15] * A[1] * A[8] - A[3] * A[12] * A[9] + A[11] * A[1] * A[12] + A[3] * A[8] * A[13]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[10] = (A[15] * A[0] * A[5] - A[15] * A[1] * A[4] - A[3] * A[12] * A[5] - A[7] * A[0] * A[13] + A[7] * A[1] * A[12] + A[3] * A[4] * A[13]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[11] = -(A[11] * A[0] * A[5] - A[11] * A[1] * A[4] - A[3] * A[8] * A[5] - A[7] * A[0] * A[9] + A[7] * A[1] * A[8] + A[3] * A[4] * A[9]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[12] = -(-A[5] * A[8] * A[14] + A[5] * A[12] * A[10] - A[12] * A[6] * A[9] - A[4] * A[13] * A[10] + A[8] * A[6] * A[13] + A[4] * A[9] * A[14]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[13] = (-A[0] * A[13] * A[10] + A[0] * A[9] * A[14] + A[13] * A[2] * A[8] + A[1] * A[12] * A[10] - A[9] * A[2] * A[12] - A[1] * A[8] * A[14]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[14] = -(A[14] * A[0] * A[5] - A[14] * A[1] * A[4] - A[2] * A[12] * A[5] - A[6] * A[0] * A[13] + A[6] * A[1] * A[12] + A[2] * A[4] * A[13]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[15] = 0.1e1 / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]) * (A[10] * A[0] * A[5] - A[10] * A[1] * A[4] - A[2] * A[8] * A[5] - A[6] * A[0] * A[9] + A[6] * A[1] * A[8] + A[2] * A[4] * A[9]);
}

/// Lower Cholesky decomposition of a square matrix.
/// No check for positive-definiteness is performed.
__device__ __host__
void cholesky(double* A, double* L, int dims){
    for ( int i = 0; i < dims*dims ; i++ )
        L[i] = 0.0 ;

    L[0] = sqrt(A[0]) ;
    for (int i = 0 ; i < dims ; i++){
        for ( int j = 0 ; j <= i ; j++){
            int ij = i + j*dims ;
            double tmp = A[ij] ;
            if ( i == j ){
                for (int k = 0 ; k < j ; k++){
                    int jk = j + k*dims ;
                    tmp -= L[jk]*L[jk] ;
                }
                L[ij] = sqrt(tmp) ;
            }
            else{
                for ( int k = 0 ; k < j ; k++){
                    int ik = i + k*dims ;
                    int jk = j + k*dims ;
                    tmp -= L[ik]*L[jk] ;
                }
                int jj = j + j*dims ;
                L[ij] = tmp/L[jj] ;
            }
        }
    }
}

__device__ __host__ void
triangular_inverse(double *L, double *Linv, int dims){
    // solve for the columns of the inverse using forward substitution
    for (int col = 0 ; col < dims ; col++ ){
        for ( int i = 0 ; i < dims ; i++ ){
            if ( i >= col ){
                double val ;
                if ( i == col )
                    val = 1 ;
                else
                    val = 0 ;

                for( int j = 0 ; j < i ; j++ )
                    val -= L[i + j*dims]*Linv[j+col*dims] ;

                Linv[i+col*dims] = val/L[i+i*dims] ;
            }
            else{
                Linv[i+col*dims] = 0.0 ;
            }
        }
    }
}

__device__ __host__ void
triangular_inverse_upper(double *U, double *Uinv, int dims){
    // solve for the columns of the inverse using backward substitution
    for (int col = dims-1 ; col >= 0  ; col-- ){
        for ( int i = dims-1 ; i >= 0  ; i-- ){
            if ( i <= col ){
                double val ;
                if ( i == col )
                    val = 1 ;
                else
                    val = 0 ;

                for( int j = dims-1 ; j > i ; j-- )
                    val -= U[i + j*dims]*Uinv[j+col*dims] ;

                Uinv[i+col*dims] = val/U[i+i*dims] ;
            }
            else{
                Uinv[i+col*dims] = 0.0 ;
            }
        }
    }
}

/// evaluate the product x*A*x'
__device__ __host__ double
quadratic_matrix_product(double* A, double *x, int dims){
    double result = 0 ;
    for ( int i = 0 ; i < dims ; i++){
        double val = 0 ;
        for ( int j = 0 ; j < dims ; j++ ){
            val += x[j]*A[i+j*dims] ;
        }
        result += x[i]*val ;
    }
    return result ;
}

//**************** Gaussian operations **********************

__device__ __host__ double
evalGaussian(Gaussian2D g, double* p){
    // distance from mean
    double d[2] ;
    d[0] = g.mean[0] - p[0] ;
    d[1] = g.mean[1] - p[1] ;

    // inverse covariance matrix
    double S_inv[4] ;
    invert_matrix<2>(g.cov,S_inv);

    // determinant of covariance matrix
    double det_S = det2(g.cov) ;

    // compute exponential
    double exponent = -0.5*quadratic_matrix_product(S_inv,d,2) ;

    return exp(exponent)/sqrt(det_S)/(2*M_PI)*g.weight ;
}

__device__ __host__ double
evalGaussian(Gaussian3D g, double* p){
    // distance from mean
    double d[3] ;
    d[0] = g.mean[0] - p[0] ;
    d[1] = g.mean[1] - p[1] ;
    d[2] = g.mean[2] - p[2] ;

    // inverse covariance matrix
    double S_inv[9] ;
    invert_matrix<3>(g.cov,S_inv);

    // determinant of covariance matrix
    double det_S = det3(g.cov) ;

    // compute exponential
    double exponent = -0.5*quadratic_matrix_product(S_inv,d,3) ;

    return exp(exponent)/sqrt(det_S)/pow(2*M_PI,1.5)*g.weight ;
}

__device__ __host__ double
evalGaussian(Gaussian4D g, double* p){
    // distance from mean
    double d[4] ;
    d[0] = g.mean[0] - p[0] ;
    d[1] = g.mean[1] - p[1] ;
    d[2] = g.mean[2] - p[2] ;
    d[3] = g.mean[3] - p[3] ;

    // inverse covariance matrix
    double S_inv[16] ;
    invert_matrix<4>(g.cov,S_inv);

    // determinant of covariance matrix
    double det_S = det4(g.cov) ;

    // compute exponential
    double exponent = -0.5*quadratic_matrix_product(S_inv,d,4) ;

    return exp(exponent)/sqrt(det_S)/pow(2*M_PI,2.0)*g.weight ;
}

template <int N, int N2>
__device__ __host__  double
computeMahalDist(Gaussian<N,N2> a, Gaussian<N,N2> b){
    // innovation vector
    double innov[N] ;
    for ( int i = 0 ; i < N ; i++ )
        innov[i] = a.mean[i] - b.mean[i] ;

    // innovation covariance
    double sigma[N2] ;

    for (int i = 0 ; i < N2 ; i++)
        sigma[i] = a.cov[i] + b.cov[i] ;

    double L[N2] ;
    cholesky(sigma,L,N);

    double Linv[N2] ;
    triangular_inverse(L,Linv,N) ;


    // multiply innovation with inverse L
    // distance is sum of squares
    double dist = 0 ;
    for ( int i = 0 ; i < N ; i++ ){
        double sum = 0 ;
        for ( int j = 0 ; j <= i ; j++){
            sum += innov[j]*Linv[i+j*N] ;
        }
        dist += sum*sum ;
    }
    return dist ;
}

template <int N, int M,int N2=N*N, int M2=M*M, int MN=M*N>
__device__ __host__ Gaussian<N,N2>
ekfUpdate(Gaussian<N,N2> predicted,double* innovation,double* H,double* R){
    // S = HPH^T
    double S[M2] ;
    double HP[MN] ;
    double Ht[MN] ;

    transpose<M,N>(H,Ht) ;
    matmultiply<M,N,N>(H,predicted.cov,HP) ;
    matmultiply<M,N,M>(HP,Ht,S) ;

    // K = P*H^T*S^-1
    double K[MN] ;
    double SInv[M2] ;
    invert_matrix<M>(S,SInv) ;
    matmultiply<N,N,M>(predicted.cov,Ht,HP) ;
    matmultiply<N,M,M>(HP,SInv,K) ;

    Gaussian<N,N2> updated ;
    // xupdate = xpredicted + K*v
    for (int i = 0 ; i < N ; i++){
        updated.mean[i] = predicted.mean[i] ;
        for (int m = 0 ; m < M ; m++){
            updated.mean[i] += innovation[m]*K[i+m*N] ;
        }
    }

    // Pupdate = (I-KH)P
    double KH[N2] ;
    matmultiply<N,M,N>(K,H,KH) ;
    for ( int i = 0 ; i < N ; i++){
        KH[i + i*N] = 1-KH[i + i*N] ;
    }
    matmultiply<N,N,N>(KH,predicted.cov,updated.cov) ;

    return updated ;
}

/// device function for summations by parallel reduction in shared memory
/*!
  * Implementation based on NVIDIA whitepaper found at:
  * http://developer.download.nvidia.com/compute/cuda/1_1/Website/projects/reduction/doc/reduction.pdf
  *
  * Result is stored in sdata[0]
  \param sdata pointer to shared memory array
  \param mySum summand loaded by the thread
  \param tid thread index
  */
__device__ void
sumByReduction( volatile double* sdata, double mySum, const unsigned int tid )
{
    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads();
    if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads();

    if (tid < 32)
    {
        sdata[tid] = mySum = mySum + sdata[tid + 32];
        sdata[tid] = mySum = mySum + sdata[tid + 16];
        sdata[tid] = mySum = mySum + sdata[tid +  8];
        sdata[tid] = mySum = mySum + sdata[tid +  4];
        sdata[tid] = mySum = mySum + sdata[tid +  2];
        sdata[tid] = mySum = mySum + sdata[tid +  1];
    }
    __syncthreads() ;
}

#endif // MATH_H
