A.
Con la garantía de sincronización de un warp un BLOCK_SIZE menor o igual a 5 funcionara.
Sin la garantía de sincronización de un warp solo funcionara un BLOCK_SIZE igual a 1.

B.
Para garantizar el orden de lectura y escritura de memoria compartida en los threads del kernel
se debe colocar una llamada a __syncthreads() entre las líneas que leen y escriben la matriz 
de memoria compartida.
