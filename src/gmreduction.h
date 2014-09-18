#ifndef GMREDUCTION_H
#define GMREDUCTION_H

#include <vector>
#include <list>
#include <algorithm>
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
vector<G> reduceGaussianMixture(vector<G> gm, double minWeight, double minDist){
//    int verbosity_ = 3 ;

    // copy gaussian vector to linked list
    list<G> pruned(gm.begin(),gm.end()) ;
//    DEBUG_VAL(pruned.size()) ;

    // prune terms with low weight
    weightPredicate<G> prunePred(minWeight) ;
//    DEBUG_MSG("perform pruning") ;
    pruned.remove_if(prunePred) ;
//    DEBUG_VAL(pruned.size()) ;

    // sort terms by descending weight
    pruned.sort(weightSortPredicate<G>);

    typename list<G>::iterator it ;
    vector<G> merged ;
    while(pruned.size() > 0){
        G maxTerm = pruned.front();
        pruned.pop_front();
        list<G> toMerge ;
        toMerge.push_back(maxTerm) ;
//        for (it = pruned.begin() ; it != pruned.end() ; it++)?
        it = pruned.begin() ;
        while( it != pruned.end()){
            double dist = computeMahalDist(maxTerm,*it) ;
            if (dist < minDist){
                toMerge.push_back(*it);
                it = pruned.erase(it) ;
            }
            it++ ;
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
