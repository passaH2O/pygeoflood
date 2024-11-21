#!/bin/bash

# activate env with TauDEM
# revise this section if you use conda, mamba, venv, etc
eval "$(micromamba shell hook --shell bash)"
micromamba activate gis-env

# number of processes, DEM name, project directory
# revise as necessary
num_proc=8
dem_name="OC1mTest"
proj_dir="data"

# TauDEM commands to obtain Dinf HAND raster
mpiexec -n $num_proc pitremove \
    -z ${proj_dir}/${dem_name}.tif \
    -fel ${proj_dir}/${dem_name}_fel.tif
mpiexec -n $num_proc dinfflowdir \
    -fel ${proj_dir}/${dem_name}_fel.tif \
    -ang ${proj_dir}/${dem_name}_ang.tif \
    -slp ${proj_dir}/${dem_name}_slp.tif
mpiexec -n $num_proc dinfdistdown \
    -fel ${proj_dir}/${dem_name}_fel.tif \
    -ang ${proj_dir}/${dem_name}_ang.tif \
    -slp ${proj_dir}/${dem_name}_slp.tif \
    -src ${proj_dir}/${dem_name}_channel_network_raster.tif \
    -dd ${proj_dir}/${dem_name}_HAND_taudem.tif \
    -m ave v
