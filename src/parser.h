#ifndef PARSER_H
#define PARSER_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <cuda_runtime.h>

using namespace std ;

vector<vector<vector <double> > > parse_datafile(const char* filename){
    string line ;
    float x ;
    float y ;
    vector<vector<vector <double> > > array ;
    ifstream file(filename) ;
    if (file.is_open()){
        while(getline(file,line)){
            vector< vector<double> > linearray(2) ;
            stringstream ss(line) ;
            while( ss.good() ){
                ss >> x ;
                ss >> y ;
                linearray[0].push_back(x) ;
                linearray[1].push_back(y) ;
            }
            array.push_back(linearray);
        }
    }
    return array ;
}

#endif // PARSER_H
