/* 
 * Code snippet for importing / exporting image data.
 * 
 * To convert an image to a pixel map, run `convert <name>.<extension> <name>.ppm
 * 
 */
/* Note that on some setups, the GPU may take more time to finish when
first invoked – you can avoid this by calling the function once before
actually timing it
 */

#include <cstdint>      // Data types
#include <iostream>     // File operations

// #define M 512       // Lenna width
// #define N 512       // Lenna height
#define M 960       // VR width
#define N 1280       // VR height
#define C 3         // Colors
#define OFFSET 16   // Header length


uint8_t* get_image_array(void){
    /*
     * Get the data of an (RGB) image as a 1D array.
     * 
     * Returns: Flattened image array.
     * 
     * Noets:
     *  - Images data is flattened per color, column, row.
     *  - The first 3 data elements are the RGB components
     *  - The first 3*M data elements represent the firts row of the image
     *  - For example, r_{0,0}, g_{0,0}, b_{0,0}, ..., b_{0,M}, r_{1,0}, ..., b_{b,M}, ..., b_{N,M}
     * 
     */        
    // Try opening the file
    FILE *imageFile;
    imageFile=fopen("./anna.ppm","rb");
    if(imageFile==NULL){
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }
    
    // Initialize empty image array
    //M*N*C is totaal aantal bits
    uint8_t* image_array = (uint8_t*)malloc(M*N*C*sizeof(uint8_t)+OFFSET);
    
    // Read the image
    fread(image_array, sizeof(uint8_t), M*N*C*sizeof(uint8_t)+OFFSET, imageFile);
    
    // Close the file
    fclose(imageFile);
        
    // Move the starting pointer and return the flattened image array
    return image_array + OFFSET;
}


void save_image_array(uint8_t* image_array){
    /*
     * Save the data of an (RGB) image as a pixel map.
     * 
     * Parameters:
     *  - param1: The data of an (RGB) image as a 1D array
     * 
     */            
    // Try opening the file
    FILE *imageFile;
    imageFile=fopen("./zllz.ppm","wb");
    if(imageFile==NULL){
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }
    
    // Configure the file
    fprintf(imageFile,"P6\n");               // P6 filetype
    fprintf(imageFile,"%d %d\n", M, N);      // dimensions
    fprintf(imageFile,"255\n");              // Max pixel
    
    // Write the image
    fwrite(image_array, 1, M*N*C, imageFile);
    
    // Close the file
    fclose(imageFile);
}

/* __global__ void inverseKleur(uint8_t* image, uint8_t* resultImage, int arraysize){
    //specifiek kleur pakken door threadID +... maal 3 (of +1 of +2) afh van kleur dat je wilt
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < arraysize; i += stride) {
        resultImage[i] = 255 - image[i] ;
    }
    } */

__global__ void inverseKleurR(uint8_t* image, uint8_t* resultImage, int arraysize){
    //specifiek kleur pakken door threadID +... maal 3 (of +1 of +2) afh van kleur dat je wilt
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < arraysize; i += stride) {
        if(i%3 == 0 && i + 6 < arraysize && i-6>0){
            resultImage[i] = 0.1*image[i-2*3] + 0.25*image[i-1*3] + 0.5*image[i] + 0.25*image[i+1*3]+0.1*image[i+2*3] ;
        }
        else{
            resultImage[i] = 255 - image[i];
        }
        //FOUT!!! 
        /* 
        //introduceert race conditie, thread 3 inverteerd een rode maar thread 1 probeert deze dan te cancellen
        int r = i*3;
        if(r+2<arraysize && i%3 == 0){
            resultImage[r] = 255 - image[r]; //vorige operatie ontdoen
            resultImage[r] = 0.1*image[r-2*3] + 0.25*image[r-1*3] + 0.5*image[r] + 0.25*image[r+1*3]+0.1*image[r+2*3] ;
        } */
    }
}

__global__ void sortKleur(uint8_t* image, uint8_t* resultImage, int arraysize){
    //specifiek kleur pakken door threadID +... maal 3 (of +1 of +2) afh van kleur dat je wilt
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i =< arraysize/3; i += stride) {
        // arraysize/3 => kan niet out of bounds gaan
        // er gaan altijd evenveel kleuren zijn
        resultImage[i] = image[i*3];
        // voor groen met offset van arraysize/3
        // rood is altijd de 1ste pixel, dus als i max is
        // zal je niet out of bounds gaan
        resultImage[i+arraysize/3] = image[i*3+1];
        //idem voor blauw
        resultImage[i+2*arraysize/3] = image[i*3+2];
    }
    }

int main (void) {
    int threads = 1024;
    int totaalThreads = N*M*C / 16;     // verhouding van 1 thread per 16 elementen, nmc grootte orde miljoen
    //int blocks = totaalThreads/threads; // totaal aantal threads verdeelt per 1024
    int blocks = 2;
    // Read the image
    uint8_t* image_array = get_image_array();    
    // Allocate output
    uint8_t* new_image_array = (uint8_t*)malloc(M*N*C);

    // Device memory
    uint8_t *d_image = nullptr;     //pointer naar device (gpu)
    uint8_t *d_new_image = nullptr; //pointer naar device (gpu)


    // Convert to grayscale using only the red color component
    // / is een gehele deling, dus als image array 0<i<3 worden in de new_image gestoken
    // i=0 is  i/3 is 0 dus new image krijgt op positie 0 de value van rood (dit is ook rood voor de nieuwe array)
    // i = 1 is 1/3 is nul dus voor groen krijgt de nieuwe array ook de waarde van rood
    // i=2 steekt de waarde van rood voor deze pixel ook in de 3de pixel (groen)
    //i=3 => 1ste element van de image_array maar *3 dus 3de element dit is opnieuw rood
/*     for(int i=0; i<M*N*C; i++){
        new_image_array[i] = image_array[i/3*3];
    } */
    float gemTijd = 0;
    cudaMalloc((void**)&d_image, M*N*C*sizeof(uint8_t));
    cudaMalloc((void**)&d_new_image, M*N*C*sizeof(uint8_t));
    cudaMemcpy(d_image, image_array, M*N*C*sizeof(uint8_t), cudaMemcpyHostToDevice);
    //hoeft niet in de loop

    cudaEvent_t start_cuda, stop_cuda;
    cudaEventCreate(&start_cuda);
    cudaEventCreate(&stop_cuda);
    cudaEventRecord(start_cuda);

    inverseKleurR<<<blocks, threads>>>(d_image, d_new_image, M*N*C);

    cudaMemcpy(new_image_array, d_new_image, M*N*C*sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_cuda);
    cudaEventSynchronize(stop_cuda);
    float ms;
    cudaEventElapsedTime(&ms, start_cuda, stop_cuda);
    gemTijd += ms;
    //memory free is niet echt overhead dus niet mee in timing.

    cudaFree(d_image); cudaFree(d_new_image);
    // Save the image
    save_image_array(new_image_array);

    gemTijd = gemTijd / 100;
    std::cout << "tijd voor " << threads <<"threads: "<< gemTijd <<std::endl;
    threads +=1; 
    
    return 0;
}
