import numpy as np
import os
import shutil 
import datetime

root_path = "/home/henriasv/simulation_data/simple_fracture_simulations_" + str(datetime.datetime.now().date())
smoothings = [150]
rbreaks = [1.135, 1.15, 1.165, 1.18, 1.21, 1.25]
ks = [1.0, 2.0, 4.0]

def create_submission_script(smoothing, rbreak, k, machine="local"):

    if machine == "local":
        out_string = """#!/bin/bash
mpirun -np 16 /home/henriasv/repos/lammps_2019_05/src/lmp_mpi -in crack.in %s 
""" % var_string
    return out_string

# CPU string
# mpirun -np 128 lmp_intel_cpu_intelmpi -pk intel 0 -sf intel -in shear.in %s
# GPU string
# mpirun -np 2 lmp_kokkos_cuda_openmpi -k on g 1 -sf kk -pk kokkos newton on -in shear.in %s

# create simulation folders 

makefile_strings = []

for smoothing in smoothings:
    for rbreak in rbreaks: 
        for k in ks:
            path = os.path.join(root_path, "smoothing_"+str(smoothing)+"_rbreak_"+str(rbreak)+"_k_"+str(k))
            os.makedirs(path)
        
            # job scipt
            with open(os.path.join(path, "job.sh"), "w") as ofile:
                var_string = "-var theta %f -var rcrit %f -var k2 %f" % (smoothing, rbreak, k)
                ofile.write(create_submission_script(smoothing, rbreak, k))

            # lammps script
            shutil.copy("lammps/crack.in", os.path.join(path, "crack.in"))
            makefile_strings.append("")
