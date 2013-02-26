#include "cutil_inline.h"

//#define BLOCK_SIZE 32
//#define GRID_SIZE 32

#define ITERS 10

typedef struct
{
  char* name
  int product_num;
  float price;
  float m_cost;
} product;

typedef struct
{
    int loyalty;
    float income;
    float bank_balance;
    int* product_req;
} consumer;

typedef struct
{
    int num;
    float bank_balance;
    product** products;
} manufacturer;




//__device__ float d_avg_of_nine(float* data_in, unsigned int x, 
//		  unsigned int y, unsigned int width) {
//}



/*__global__ void device_blur_old(float* data_in, float* data_out) {
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  const unsigned int y = tid+1;
  const unsigned int stride = y*PADWIDTH;

  extern __shared__ float row[];
  for (int i = stride; i < (stride+PADWIDTH); i++) {
    row[i] = data_in[i];
  }
  __syncthreads();

  #pragma unroll
  for (int x = 1; x <= MATRIX_WIDTH; x++) {
    //for (int y = 1; y <= MATRIX_HEIGHT; y++) {
      data_out[x+stride] = d_avg_of_nine(row, x, y, PADWIDTH);
      //}
  }
  }  */

void copy_array(float* from, float* to, unsigned int size) {
  for (int i = 0; i < size; i++) {
    to[i] = from[i];
  }
}

void print_array(float* data_in, unsigned int size) {
  for (int i=0; i < size; i++) {
    printf("%f,", data_in[i]);
  }
  printf("\n\n");
}

double sum_array(float* data_in, unsigned int length) {
  double rezult = 0;
  for (int k=0; k < length; k++)
    rezult += data_in[k];
  return rezult;
}



/* First arg: threads per block,
   Second arg: blocks per grid */
int main(int argc, char** argv)
{
  if (argc != 3) {
    printf("Please input two arguments: threads and blocks\n");
    exit(1);
  }

  int threadsPerBlock = atoi(argv[1]);
  int blocksPerGrid = atoi(argv[2]);

  int devID;
  cudaDeviceProp props;

  // get number of SMs on this GPU
  cutilSafeCall(cudaGetDevice(&devID));
  cutilSafeCall(cudaGetDeviceProperties(&props, devID));



  





  // allocate host memory 
  /*  unsigned int mem_size = sizeof(float) * PADWIDTH*PADHEIGHT;
  float* h_data_in      = (float*)calloc(sizeof(float), mem_size);
  float* h_data_out     = (float*)malloc(mem_size);

  printf("Input size : x:%d, y:%d\n", MATRIX_WIDTH, MATRIX_HEIGHT);
  //printf("Grid size  : %d\n", GRID_SIZE);
  //  printf("Block size : %d\n", BLOCK_SIZE);

  printf("Grid size  : %d\n", blocksPerGrid);
    printf("Block size : %d\n", threadsPerBlock);


  datainit(h_data_in);

  // allocate device memory
  float* d_data_in;
  cutilSafeCall(cudaMalloc((void**) &d_data_in, mem_size));
  float* d_data_out;
  cutilSafeCall(cudaMalloc((void**) &d_data_out, mem_size));

  cutilSafeCall(cudaMemcpy(d_data_in, h_data_in, 
			   mem_size, cudaMemcpyHostToDevice));

  unsigned int globalOffset = 0;
  unsigned int rowOffset = globalOffset % MATRIX_WIDTH;
  
  // set up kernel for execution
  unsigned int timerd = 0;
  cutilCheckError(cutCreateTimer(&timerd));
  cutilCheckError(cutStartTimer(timerd));  
  
  //printf("For loop: \n");
  for (int j = 0; j < ITERS; j++) {
    //printf("%d, ",j);
    cutilSafeCall(cudaMemset(d_data_out, 0, mem_size));
   
    //int shmemSize = PADWIDTH*threadsPerBlock*sizeof(float);

    device_blur<<<blocksPerGrid, threadsPerBlock>>>(d_data_in, d_data_out, globalOffset, rowOffset);
    cudaThreadSynchronize();
    
    //cutilSafeCall(cudaFree(d_data_in));
  }
  
  // check if kernel execution generated an error
  cutilCheckMsg("Kernel execution failed");

  // stop and destroy timer
  cutilCheckError(cutStopTimer(timerd));
  double dSeconds = cutGetTimerValue(timerd)/(1000.0);
  double dNumOps = ITERS * MATRIX_WIDTH * MATRIX_HEIGHT * 42;
  double gflops = dNumOps/dSeconds/1.0e9;
  double averageDevTime = dSeconds/ITERS*1000; // milliseconds

  //Log throughput
  printf("Device average exec time: %.8f milliseconds\n", averageDevTime);
  printf("Throughput = %.4f GFlop/s\n", gflops);
  cutilCheckError(cutDeleteTimer(timerd));

  cutilSafeCall(cudaMemcpy(h_data_out, d_data_out, 
			   mem_size, cudaMemcpyDeviceToHost));

  printf("Run %d Kernels.\n\n", ITERS);
  unsigned int timer = 0;
  cutilCheckError(cutCreateTimer(&timer));
  cutilCheckError(cutStartTimer(timer));  
  
  float* host_blur_data_out = (float*)calloc(sizeof(float),mem_size);  

  for (int j = 0; j < ITERS; j++) {
    host_blur(h_data_in, host_blur_data_out);
  }

  //printf("Before calloc\n");
//  float* fake_blur_data_out = (float*)calloc(sizeof(float),mem_size);  
//  printf("After calloc\n");
//  for (int j = 0; j < ITERS; j++) {
//    fake_device_blur(h_data_in, fake_blur_data_out, 0, 0);
//    }


  // check if kernel execution generated an error
  cutilCheckMsg("Kernel execution failed");

  // stop and destroy timer
  cutilCheckError(cutStopTimer(timer));
  double hSeconds = cutGetTimerValue(timer)/(1000.0);
  double averageTime = hSeconds/ITERS; // milliseconds
  printf("Host average exec time: %.8f milliseconds\n", averageTime*1000);

  double sumOfHostBlur = sum_array(host_blur_data_out, PADWIDTH*PADHEIGHT);
  double sumOfDevBlur = sum_array(h_data_out, PADWIDTH*PADHEIGHT);
  printf("HostBlur result: %.4f\n", sumOfHostBlur);
  printf("DeviceBlur result: %.4f\n", sumOfDevBlur);

  //double sumOfFakeBlur = sum_array(fake_blur_data_out, PADWIDTH*PADHEIGHT);
  //printf("FakeBlur result: %.4f\n", sumOfFakeBlur);

  printf("IN MATRIX:\n");
  print_matrix(h_data_in);
  printf("DEV OUT MATRIX:\n");
  print_matrix(h_data_out);
  printf("HOST OUT MATRIX:\n");
  print_matrix(host_blur_data_out);
  printf("FAKE OUT MATRIX:\n");
  print_matrix(fake_blur_data_out);

  // clean up memory
  free(h_data_in);
  free(h_data_out);
  free(host_blur_data_out);
  //free(fake_blur_data_out);
  
  // exit and clean up device status
  cudaThreadExit();*/
}
