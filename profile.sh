#!/usr/bin/bash

DATAFILE=data2.txt
CONFIGFILE=config.json
NPARTICLES=(50 100 200 500 1000 2000 5000 10000)

# clear the results file
rm times.log

# loop over number of particles
for n in "${NPARTICLES[@]}" ; do
    # edit the config file
    sed -i "s/\(\"nParticles\" : \)[0-9]*,/\1${n},/" $CONFIGFILE
    
    # first column in results file is nParticles
    printf "%d " "${n}" >> times.log
    
    # 10 repetitions
    for i in `seq 1 10`; do
      # time before doing work, concatentated with nanoseconds
      T="$(date +%s%N)"
      # launch
      ./scphd-cuda $CONFIGFILE $DATAFILE
      # subtract initial timestamp from current time
      T="$(($(date +%s%N)-T))"
      
      # convert to milliseconds
      M="$((T/1000000))"
      
      # log time
      printf "%f " "${M}" >> times.log
    done
    
    # line break
    printf "\n" >> times.log
done
