#include <iostream>
#include <cuda_runtime.h>
//nu met meerdere blocks
// CUDA kernel to rev 1 arrays
__global__ void rev(int *a, int *c, int n, int arraysize) { 
    //index wordt n-i want omgekeerd doorlopen
    // iedere element heeft zijn eigen thread in 1 blok, dus index i komt overeen met threadId
    // cores binnen 1 sm delen de ram? dus kunnen ze aan elkaars element?
    int idx = threadIdx.x  + blockIdx.x * blockDim.x;
    //threadId begint vanaf nul per blok
    //dus om index te vinden moet je het blokID (begint vanaf 0) verminigvuldigen met
    //hoeveel threads er in een blok zitten (de dimensie (1 dimensionaal voor array))
    //In de main wordt er voor gezorgt dat er niet teveel (of te weinig) thread Ids + blockId*blockdim zijn
    if (idx < arraysize) {
        c[arraysize-idx-1] = a[idx]; //moet min 1 want index gaat van 0 tot 8
    }
}

int main() {
    //alles hier gaat over de gpu, alleen de rev wordt opgeroepen en dit is een normale code
    //er zijn hier geen branches, dit is de bedoeling, conditionele sprongen worden genomen in de rev maar dit maakt niet uit want
    // dit wrodt toch op de cpu gedaan
    for (int block = 997; block < 1000; block++){ //block nul kan niet dus beginnen van 1, vanaf 9 blocks is de rest redundant
        cudaEvent_t start_cuda, stop_cuda;
        cudaEventCreate(&start_cuda);
        cudaEventCreate(&stop_cuda);
        cudaEventRecord(start_cuda);
        const int N = 1000; //van nul tot acht is array grootte maar ook aantal threads 
        //const int threads = N/block; //want blocks maal threads moet gelijk zijn aan de array size om logisch te zijn.
        const int threads = N/block;
        //const int block = 3;
        //int grootte = 9;
        //if (N != grootte){ N = grootte;} //evenveel threads als elementen in array
        int a[N];
        int c[N]; //resultaat
        int *d_a; //pointer naar begin array
        int *d_c;
        //if(threads*block == N){
        // Initialize host arrays
        //std::cout << "begin array ";
        
        for (int i = 0; i < N; i++) {
            a[i] = i;
            c[i] = 0;
            //std::cout << a[i] << " ";

        }
        std::cout << std::endl; 

        std::cout << "blocks " << block << std::endl;
        std::cout << "threads " << threads << std::endl;


        // Allocate device memory
        cudaMalloc((void**)&d_a, N * sizeof(int));
        cudaMalloc((void**)&d_c, N * sizeof(int));

        
        // Copy data from host to device
        cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
        //moet niet voor c denk ik want is toch lege array?   
        // Launch kernel (1 block of N threads)
        rev<<<block, threads>>>(d_a, d_c, threads, N);

        // Copy result back to host
        cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();
        cudaEventRecord(stop_cuda);
        cudaEventSynchronize(stop_cuda);
        float ms;
        cudaEventElapsedTime(&ms, start_cuda, stop_cuda);
        std::cout << "tijd: "<<ms << std::endl;


        // Print results
/*         std::cout << "Result: ";
        for (int i = 0; i < N; i++) {
            std::cout << c[i] << " ";
        }
        std::cout << std::endl;  */

        // Free device memory
        cudaFree(d_a);
        cudaFree(d_c);
        std::cout << std::endl << std::endl;

}
    return 0;
}
