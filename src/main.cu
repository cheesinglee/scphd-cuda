#include <vector>
#include <list>
#include <algorithm>
#include <string>
#include <ctime>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include "models.cuh"
#include "scphd.cuh"
#include "parser.h"
#include "math.h"
#include "json/json.h"
#include "json/json-forwards.h"

using namespace std;




int main(int argc, char* argv[])
{
    if (argc < 3){
        cout << "Usage: scphd-cuda [configuration] [data]" << endl;
        exit(0);
    }


    Json::Value root ;
    Json::Reader reader ;
    ifstream configFilename(argv[1]) ;
    bool success = reader.parse(configFilename,root) ;
    if (!success){
        // report to the user the failure and their locations in the document.
        std::cout  << "Failed to parse configuration\n"
                   << reader.getFormattedErrorMessages();
        exit(0);
    }
    int nParticles = root["nParticles"].asInt() ;
    double ps = root["ps"].asDouble() ;
    double pd = root["pd"].asDouble() ;
    double w0 = root["w0"].asDouble() ;
    double kappa = root["kappa"].asDouble() ;
    double pruneWeight = root["pruneWeight"].asDouble() ;
    double mergeDist = root["mergeDist"].asDouble() ;

    double sigma[2] = {1.0,1.0} ;
    double R[4] = {1.0,0.0,0.0,1.0} ;
    BrownianMotionParentModel<2> parentModel(sigma) ;
    BrownianMotionDaughterModel<2,Gaussian2D> daughterModel(sigma) ;
    BiasedIdentityModel<2> measurementModel(R,0.98) ;

    // seed RNG - don't forget this!
    srand(time(NULL)) ;

    // check cuda device properties
    int nDevices ;
    cudaGetDeviceCount( &nDevices ) ;
    cout << "Found " << nDevices << " CUDA Devices" << endl ;
    cudaDeviceProp props ;
    cudaGetDeviceProperties( &props, 0 ) ;
    cout << "Device name: " << props.name << endl ;
    cout << "Compute capability: " << props.major << "." << props.minor << endl ;

    SCPHDFilter<2,Gaussian2D,2,BrownianMotionParentModel<2>,BrownianMotionDaughterModel<2,Gaussian2D>,BiasedIdentityModel<2>>
    filter(nParticles,ps,pd,w0,kappa,pruneWeight,mergeDist,
           parentModel,daughterModel,measurementModel) ;


    vector<vector < vector<double> > > measurements = parse_datafile(argv[2]) ;
    int nSteps = measurements.size() ;
    cout << nSteps << endl ;
    vector<Gaussian2D> est ;
    for (int k = 0 ; k < nSteps ; k++){
        cout << "k = " << k << endl ;
        vector < vector<double> > Z = measurements[k] ;
//        cout << "predict" << endl ;
        filter.predict();
//        cout << "update" << endl ;
        filter.update(Z);
//        cout << "resample" << endl ;
        est = filter.daughterEstimate() ;
        filter.resample();
    }
    vector<Gaussian2D>::iterator it ;
    for (it = est.begin() ; it != est.end() ; it++){
        cout << "(" << it->mean[0] << "," << it->mean[1] << ")" << endl ;
    }


    return 0;
}

