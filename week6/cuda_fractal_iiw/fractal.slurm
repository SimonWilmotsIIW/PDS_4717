#!/bin/bash -l
#SBATCH --cluster=genius
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=lp_h_pds_iiw
#SBATCH --time=00:05:00
#SBATCH --error="%x.e%A"
#SBATCH --output="%x.o%A"

source /data/leuven/303/vsc30380/slurmhooks

module purge
module load GCC/10.3.0
module load CUDA/11.7.0

nvcc -O3 -DNDEBUG -o fractal fractal.cu main.cpp image2d.cpp -gencode arch=compute_60,code=sm_60

rm -f output1.png
rm -f output2.png
rm -f output3.png
./fractal output1 255 -2 1 -1 1
./fractal output2 5000 -0.75914 -0.72692 0.115693 0.137173
./fractal output3 1000 -0.7458547 -0.7412791 0.1298771 0.1329275
./fractal output4 400 -0.76006598207706 -0.76007566630506  0.08049577394589 0.08048608971789
./fractal output5 400 -0.76272491751166 -0.76303941125246  0.0828469893113 0.0825324955705
./fractal output6 400 -0.11031108087207 -0.10715091471597 -0.89384934408182i -0.89577030011382
