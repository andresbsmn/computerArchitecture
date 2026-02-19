#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <chrono>
//nu met atomic operation    
struct Result{
    int hoogste;
    int pos;
};

__global__ void max(int* a, int* dmax, int size){
    int idx = threadIdx.x  + blockIdx.x * blockDim.x;
    if(idx < size){
        atomicMax(dmax,a[idx]); 
        //1st pointer naar plaats waar maximum wordt opgeslagen, dan wordt rechter val vergeleken
        //met waarde die in de pointer staat en de hoogste van de twee wordt daar opgeslaan
        //lijkt erop dat dit niets returned om te weten of je een nieuwe hoogste value hebt
        //dus zou moeten vergelijken met een 2de var om te zien of deze is verandert en op deze manier
        //de threadidx gebruiken om de positie te vinden, kan later nog toegevoegd worden.
        //nu dus niet via result werken maar gewoon een int, als terug met result haal uit commentaar en vervang
        //in max(int => result) en ook bij terugcopieren 
    }
}


int main() {
    int *p_arr;
    //int block = 1; //memory is shared per block, start with one for ease
    //Result result;
    int* dmax;
    float tijdgem;
    int aantalcycli = 1000;
    int maxval;
    srand(67); //seed for consistant results
    for(int N = 1000; N < 1000000; N += 100000){
        //max threads per block 1024
        //dus array kleiner dan 1024 als we per element een thread willen in 1 block
        //blocks maal threads moet even groot zijn als array 
        //moeilijk om te voorspellen dus kiezen we een vaste waarde voor de threads
        int a[N];
        int threads = 250;
        int block = (N+threads - 1)/threads; //https://stackoverflow.com/a/2745086
        tijdgem = 0;
        for (int i = 0; i < N; i++) {
            a[i] = rand() % 101; // rand() % 101 gives 0 to 100
        }
/*     for  (int i = 0; i < N; i++) {
        std::cout << a[i] << " ";
    } */
    //std::cout << std::endl;
        for(int i = 0; i<aantalcycli; i++){
            maxval = 0;


            cudaMalloc((void**)&p_arr, N * sizeof(int));
            //cudaMalloc((void**)&result, sizeof(result));
            cudaMalloc((void**)&dmax, sizeof(int));
        //copy to gpu
            cudaMemcpy(p_arr, a, N * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(dmax, &maxval, sizeof(int), cudaMemcpyHostToDevice);
            cudaEvent_t start_cuda, stop_cuda;
            cudaEventCreate(&start_cuda);
            cudaEventCreate(&stop_cuda);
            cudaEventRecord(start_cuda);
            max<<<block, threads>>>(p_arr, dmax, N);
            cudaDeviceSynchronize();
            cudaEventRecord(stop_cuda);
            cudaEventSynchronize(stop_cuda);
            float ms;
            cudaEventElapsedTime(&ms, start_cuda, stop_cuda);
        //copy back to host
            cudaMemcpy(&maxval, dmax, sizeof(int), cudaMemcpyDeviceToHost);
        //overhead mee in timing   
            
            tijdgem += ms;
            //std::cout << "debug: "<< tijdgem << std::endl;
            cudaFree(p_arr);
            cudaFree(dmax);
            }

        tijdgem = tijdgem/aantalcycli;
        std::cout << "array met grootte "<< N << " heeft gemiddelde tijd: "<<tijdgem <<  "ms"<< std::endl;
        //std::cout << "hoogste is " << result.hoogste << " op positie " << result.pos << std::endl; 
        //std::cout << "hoogste is: "<< maxval << std::endl;
}
    return 0;
}
