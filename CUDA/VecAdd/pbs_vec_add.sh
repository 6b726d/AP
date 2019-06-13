#!/bin/sh
#PBS -N Vec_Add
#PBS -l nodes=1:ppn=1:gpus=1

cd $PBS_O_WORKDIR

./vec_add
