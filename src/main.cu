#include <vector>
#include <list>
#include <algorithm>
#include <string>
#include <ctime>
#include <iostream>
#include <sstream>
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



    string sigmaStr = root["sigmaparent"].asString() ;
    stringstream sigmaSS(sigmaStr) ;
    double sigma[2] ;
    sigmaSS >> sigma[0] >> sigma[1] ;
    BrownianMotionParentModel<2> parentModel(sigma) ;

    sigmaStr = root["sigmadaughter"].asString() ;
    sigmaSS.str(sigmaStr) ;
    sigmaSS >> sigma[0] >> sigma[1] ;
    BrownianMotionDaughterModel<2,Gaussian2D> daughterModel(sigma) ;


    string Rstr = root["R"].asString() ;
    stringstream Rss(Rstr) ;
    double R[4] ;
    Rss >> R[0] >> R[1] >> R[2] >> R[3] ;
    BiasedIdentityModel<2> measurementModel(R,pd) ;

    // seed RNG - don't forget this!
    srand(time(NULL)) ;

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
        cout << "predict" << endl ;
        filter.predict();
        cout << "update" << endl ;
        filter.update(Z);
        cout << "daughter estimate" << endl ;
        est = filter.daughterEstimate() ;

        filter.resample();
    }
    vector<Gaussian2D>::iterator it ;
    for (it = est.begin() ; it != est.end() ; it++){
        cout << "(" << it->mean[0] << "," << it->mean[1] << ")" << endl ;
    }


    return 0;
}

