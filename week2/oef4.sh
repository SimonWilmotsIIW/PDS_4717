#! /bin/bash -l
#SBATCH --cluster=wise
#SBATCH --account=lp_h_pds_iiw
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --time=00:50:00
#SBATCH --error="%x.e%A"
#SBATCH --output="%x.o%A"

source /data/leuven/303/vsc30380/slurmhooks

module purge

module load CMAKE
cmake --version


 /data/leuven/303/vsc30380/slurmquote.py
