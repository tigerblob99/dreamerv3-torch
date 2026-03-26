#! /bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:10:00
#SBATCH --partition=devel
#SBATCH --qos=basic

module load mpitest

mpirun mpihello
