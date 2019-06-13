1. Ingresar a Manati:
   manati.iiap.gob.pe
   522
2. $ /usr/local/cuda/bin/nvcc -std=c++11 -o code code.cu
3. $ nano pbs_code.sh
4. $ qsub pbs_code.sh
5. $ cat Code.o111111

Error en pbs GPU (Solucion):
$pbsnodes
Elegir el nodo a usar. Por ejemplo n009.
En pbs_code.sh:
  #!/bin/sh
  #PBS -l nodes=n009:ppn=1:gpus=1
  
  cd $PBS_O_WORKDIR
  
  ./code
