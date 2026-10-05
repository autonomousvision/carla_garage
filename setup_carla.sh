#!/usr/bin/env bash

# Download and install CARLA
mkdir carla
cd carla
wget --content-disposition https://downloads.carlasim.com/Linux/CARLA_0.9.15.tar.gz
tar -xf CARLA_0.9.15.tar.gz
cd Import
wget --content-disposition https://downloads.carlasim.com/Linux/AdditionalMaps_0.9.15.tar.gz
cd ..
./ImportAssets.sh
rm CARLA_0.9.15.tar.gz
cd ..