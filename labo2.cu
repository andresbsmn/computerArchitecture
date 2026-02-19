#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <chrono>
    
struct Result{
    int hoogste;
    int pos;
};

Result max(int a[], int size){
    int hoogste;
    int pos;
    hoogste = a[0]; //altijd beginnen bij een waarde die bestaat
    pos = 0;
    for(int i = 1; i<size; i++ ){
        if (a[i]>hoogste){
            hoogste = a[i];
            pos = i;
        }
    }
    return {hoogste, pos};
}

int main() {
    //int N = 20;
    int aantalcycli = 1000;
    float tijdgem = 0;
    Result result;
    srand(67); //seed for consistant results
    for(int N = 1000; N < 1000000; N += 100000){
    int arr[N];
    for (int i = 0; i < N; i++) {
        arr[i] = rand() % 101; // rand() % 101 gives 0 to 100
    }
/*     for (int i = 0; i < N; i++) {
        std::cout << arr[i] << " ";
    } */
    std::cout << std::endl;
    for(int i = 0; i < aantalcycli; i++){
        const auto start = std::chrono::steady_clock::now();
        result = max(arr, N);
        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed_seconds{end-start};
        tijdgem += elapsed_seconds.count()/1000;
    }
    tijdgem = tijdgem/aantalcycli;
    std::cout << "array met grootte "<< N << " heeft gemiddelde tijd: "<<tijdgem <<  "ms"<< std::endl;
    std::cout << "hoogste is " << result.hoogste << " op positie " << result.pos << std::endl; 

    }
//    std::cout << "dit duurde: " << elapsed_seconds.count() << std::endl;

    return 0;
}
