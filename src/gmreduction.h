#ifndef GMREDUCTION_H
#define GMREDUCTION_H

#include <vector>
#include <list>
#include <algorithm>
#include <nvToolsExtCuda.h>
#include "math.h"
#include "types.h"

#define DEBUG_MSG(x) if ( verbosity_ >= 2) cout << "[" << __func__ << "(" << __LINE__ << ")]: " << x << endl
#define DEBUG_VAL(x) if ( verbosity_ >= 2) cout << "[" << __func__ << "(" << __LINE__ << ")]: " << #x << " = " << x << endl

using namespace std ;

template <class G>
class weightPredicate{
public:
    double minWeight ;
    weightPredicate(double w) {minWeight = w ;}
    bool operator() (G gaussian) {return gaussian.weight < minWeight ;}
};

template <class G>
bool weightSortPredicate(G a, G b) { return a.weight >= b.weight ;}

bool isFalse(bool b) { return !b; }

//bool weightSortPredicate(Gaussian2D a, Gaussian2D b) { return (a.weight >= b.weight) ;}

template <int N, int N2 = N*N >
Gaussian<N,N2> mergeGaussians(list< Gaussian<N,N2> > gaussians){
    // initialize
    Gaussian<N,N2> merged ;
    merged.weight = 0.0 ;
    for ( int i =0 ; i < N2 ; i++){
        if (i < N)
            merged.mean[i] = 0.0 ;
        merged.cov[i] = 0.0 ;
    }
    typename list< Gaussian<N,N2> >::iterator it ;

    // merged weight and mean
    for (it = gaussians.begin() ; it != gaussians.end() ; it++){
        double w = it->weight ;
        merged.weight += w ;
        for (int i=0 ; i < N ; i++){
            merged.mean[i] += it->mean[i] * w ;
        }
    }
    for (int i=0 ; i < N ; i++){
        merged.mean[i] /= merged.weight ;
    }

    // merged covariance
    for (it = gaussians.begin() ; it != gaussians.end() ; it++){
        double innov[N] ;
        for (int i=0 ; i < N ; i++){
            innov[i] = merged.mean[i] - it->mean[i] ;
        }
        for (int i = 0 ; i < N ; i++){
            for (int j = 0 ; j < N ; j++){
                int idx = i+j*N ;
                merged.cov[idx] += it->weight * (it->cov[idx] + innov[i]*innov[j]) ;
            }
        }
    }
    for (int i=0 ; i < N2 ; i++){
        merged.cov[i] /= merged.weight ;
    }

    return merged ;
}

template <class G>
vector<G> reduceGaussianMixture(vector<G> gm,
                                double minWeight, double minDist){
//    int verbosity_ = 3 ;



//    // mark pruned features
//    for (int i = 0 ; i < N ; i++){
//        mergedFlags[i] = (gm[i].weight < minWeight) ;
//    }

    // prune and copy gaussian vector
    vector<G> pruned ;
    for ( int i = 0 ; i < gm.size() ; i++){
        G feature = gm[i] ;
        if (feature.weight > minWeight){
            pruned.push_back(feature);
        }
    }
//    DEBUG_VAL(pruned.size()) ;

    // flag array to keep track of what has been merged
    int N = pruned.size() ;
    vector<bool> mergedFlags(N,false) ;

    // compute distance matrix
    nvtxRangePushA("Compute merge distances") ;
    vector< vector<double> > distances(N) ;
    for ( int row = 0 ; row < N ; row++){
        distances[row].resize(N,0.0) ;
        for (int col = 0 ; col < N ; col++){
            if (col < row){
                G g1 = pruned[row] ;
                G g2 = pruned[col] ;
                double dist = computeMahalDist(g1,g2) ;
                distances[row][col] = dist ;
                distances[col][row] = dist ;
            }
        }
    }
    nvtxRangePop() ;



    typename list<G>::iterator it ;
    vector<G> merged ;
    while(any_of(mergedFlags.begin(),mergedFlags.end(),isFalse)){
        // find the max term
        int maxIdx = 0 ;
        double maxWeight = -1 ;
        for ( int i = 0 ; i < N ; i++){
            if (pruned[i].weight > maxWeight && !mergedFlags[i]){
                maxWeight = pruned[i].weight ;
                maxIdx = i ;
            }
        }
        G maxTerm = pruned[maxIdx] ;
        mergedFlags[maxIdx] = true ;
        list<G> toMerge ;
        toMerge.push_back(maxTerm) ;
        for (int j = 0 ; j < N ; j++){
            double dist = distances[maxIdx][j] ;
            if (dist < minDist && !mergedFlags[j]){
                toMerge.push_back(pruned[j]);
                mergedFlags[j] = true ;
            }
        }
        if (toMerge.size() == 1){
            merged.push_back(maxTerm);
        }else{
            merged.push_back(mergeGaussians(toMerge));
        }
    }
//    DEBUG_VAL(merged.size()) ;
    return merged ;
}

#endif // GMREDUCTION_H
