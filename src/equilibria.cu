#include "cutil_inline.h"

#include "math.h"
#include "time.h"
#include "stdlib.h"
#include "limits.h"
#include "stdio.h"

#define ITERS 10

// Direction of price for the given manufacturer.
// Up means prices are increasing, down is decreasing
#define STRATEGIES 2
#define STRATEGY_UP 1
#define STRATEGY_DOWN 2

//#define BLOCK_SIZE 32
//#define GRID_SIZE 32
#define NUM_MANUFACTURERS 2
#define NUM_CONSUMERS 10
#define MAX_MARGINAL 250
#define BASE_INCOME 20000
#define PRICE_INCREMENT 2

const char* products[] = {"milk", "bread", "toilet_paper", "butter", "bacon", "cheese"};
int NUM_PRODUCTS = sizeof(products)/sizeof(char*);

int select_loyalty();
double gaussrand();
double positive_gaussrand();

typedef struct
{
  char* name;
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

// First dimension is product ID
// Second dimension is manufacturer ID
// Price is in pence
int** price; // 2D array

int* marginal_cost;

// First dimension is product ID
// Second dimension is consumer ID
// Consumption is units of product consumed per day
int** consumption; // 2D array

// Array mapping consumer ID to income in pence per day
int* income;

// Array mapping consumer ID to manufacturer ID
// Showing which manufacturer this consumer currently prefers
int* loyalty; 

// Arrays mapping manufacturer ID to profit on each day
typedef struct
{
  int* two_days_ago;
  int* yesterday;
  int* today;
} profits;

profits* profit_history;

// Array mapping manufacturer ID to price strategy 
// (that is, one of the constants from STRATEGY_UP and _DOWN)
int* price_strategy;


/*
// Get the manufacturer ID fom which the consumer chooses to 
// purchase the given product
__device__ int consumer_choice(int consumer_id, int product_id) {
}

// Get tomorrow's price for the given product ID
__device__ int price_response(int manufacturer_id, int product_id) {
}

__global__ void equilibriate(int** price, int** consumption, int* income, int* loyalty, profits* profit) {
}

*/

void init_strategy() 
{
  price_strategy = (int*) malloc(NUM_MANUFACTURERS*sizeof(int));
  
  int i;
  for (i = 0; i < NUM_MANUFACTURERS; i++) 
  {
    // Randomly choose int between 1 and num of strategies
    float randVal = (float)rand()/RAND_MAX;
    price_strategy[i] = (int)(randVal*STRATEGIES) + 1;
  }
}


void init_marginal()
{
  marginal_cost = (int*) malloc(NUM_PRODUCTS * sizeof(int));

  int i;
    
  for (i = 0; i < NUM_PRODUCTS; ++i) {
    float rval = (float)rand()/RAND_MAX;
    marginal_cost[i] = (int)(rval * MAX_MARGINAL);
    printf("Marginal cost for %s is %d.\n", products[i], marginal_cost[i]);
  }
}

// Rand*MC*3 (roughly)
void init_prices()
{
  int i;
  int j;

  price = (int**) malloc(NUM_PRODUCTS * sizeof(int*));

  for (i = 0; i < NUM_PRODUCTS; ++i) {
    price[i] = (int*) malloc(NUM_MANUFACTURERS * sizeof(int));
    for (j = 0; j < NUM_MANUFACTURERS; ++j) {
      float rval = (float)rand()/RAND_MAX;
      price[i][j] = marginal_cost[i] + (rval * marginal_cost[i]);
      printf("Price for %s from manufacturer %d: %d\n", products[i], j, price[i][j]);
    }
  }
}

// Uniformly distributed
void init_loyalty()
{
  loyalty = (int*) malloc(NUM_CONSUMERS * sizeof(int));
    
  int i;
  int* counts = (int*) malloc(NUM_MANUFACTURERS * sizeof(int));
    
  for (i = 0; i < NUM_CONSUMERS; ++i) {
    loyalty[i] = select_loyalty();
    //	printf("Customer %d loyal to manufacturer %d\n", i, loyalty[i]);
    counts[loyalty[i]]++;
  }

  for (i = 0; i < NUM_MANUFACTURERS; ++i) {
    printf("Manufacturer %d has %d loyal customers.\n", i, counts[i]);
  }
}

int select_loyalty()
{
  int i;
  float rval = (float)rand()/RAND_MAX;

  float split = 1.0/NUM_MANUFACTURERS;
  for (i = 0; i < NUM_MANUFACTURERS; ++i) {
    if (rval < split * (i + 1))
      return i;
  }
  return i;
}

/*
 * Gaussian over population. Currently generates values using a gaussian tail
 * distribution - there will be a lot of people who have an income around 
 * the base income, and fewer with higher incomes.
 */
void init_income()
{
  income = (int*) malloc(NUM_CONSUMERS * sizeof(int));
    
  int i;
  for (i = 0; i < NUM_CONSUMERS; ++i) {
    income[i] = BASE_INCOME * (positive_gaussrand() + 1);
    printf("Income of household %d: %d\n", i, income[i]);
  }
}

// Initialise last two days of profits with fake values.
// All profits two days ago are set to 0 and for yesterday 
// are set to 1. Thus, all profits increase so currently
// active strategies are kept in place and acted on
void init_profits() 
{
  profit_history = (profits*) malloc(sizeof(profits));
  profit_history->two_days_ago = (int*) malloc(sizeof(int)*NUM_MANUFACTURERS);
  profit_history->yesterday = (int*) malloc(sizeof(int)*NUM_MANUFACTURERS);
  profit_history->today = (int*) malloc(sizeof(int)*NUM_MANUFACTURERS);

  int man;
  for (man = 0; man < NUM_MANUFACTURERS; ++man)
  {
    profit_history->two_days_ago[man] = 0;
    profit_history->yesterday[man] = 1;
  }
}



/* Generate a gaussian random value in the interval [0,infinity] */
double positive_gaussrand()
{
  double r;
  while ((r = gaussrand()) < 0);
  return r;
}

// Polar method implementation taken from c-faq.com/lib/gaussian.html
double gaussrand()
{
  static double V1, V2, S;
  static int phase = 0;
  double X;
    
  if (phase == 0){
    do {
      double U1 = (double)rand()/RAND_MAX;
      double U2 = (double)rand()/RAND_MAX;
	    
      V1 = 2 * U1 - 1;
      V2 = 2 * U2 - 1;
      S = V1 * V1 + V2 * V2;
    } while (S >= 1 || S == 0);
    X = V1 * sqrt(-2 * log(S) / S);
  } else {
    X = V2 * sqrt(-2 * log(S) / S);
  }

  phase = 1 - phase;
    
  return X;
}



// Get the manufacturer ID fom which the consumer chooses to 
// purchase the given product
int host_consumer_choice(int consumer_id, int product_id) {
  return 0;
}

// Get tomorrow's price for the given product ID
void host_price_response(int manufacturer_id, int product_id) {
  int current_strategy = price_strategy[manufacturer_id];
  int profit1 = profit_history->two_days_ago[manufacturer_id];
  int profit2 = profit_history->yesterday[manufacturer_id];

  // If profit decreased, switch strategy
  if (profit1 > profit2) 
  {
    if (current_strategy == STRATEGY_UP) 
    {
      price_strategy[manufacturer_id] = STRATEGY_DOWN;
    }
    else
    {
      price_strategy[manufacturer_id] = STRATEGY_UP;
    }
  }

  price[product_id][manufacturer_id] += PRICE_INCREMENT;
}

void host_equilibriate(int** price, int** consumption, int* income, int* loyalty, profits* profit) {
}

void copy_array(float* from, float* to, unsigned int size) {
  for (int i = 0; i < size; i++) {
    to[i] = from[i];
  }
}

void print_array(float* data_in, unsigned int size) {
  for (int i=0; i < size-1; i++) {
    printf("%f,", data_in[i]);
  }
  printf("%f\n\n", data_in[size-1]);
}

void print_int_array(int* data_in, unsigned int size) {
  for (int i=0; i < size-1; i++) {
    printf("%d,", data_in[i]);
  }
  printf("%d\n\n", data_in[size-1]);
}

double sum_array(float* data_in, unsigned int length) {
  double rezult = 0;
  for (int k=0; k < length; k++)
    rezult += data_in[k];
  return rezult;
}




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

  // ---------------------------------------------

  srand(time(NULL));

  /* int i; */
  /* for (i = 0; i < 100; ++i) { */
  /*   printf("%lf\n", positive_gaussrand() + 1); */
  /* } */


  init_income();
  init_loyalty();
  init_marginal();
  init_prices();

  init_strategy();
  init_profits();
  printf("Printing two days ago\n");
  print_int_array(profit_history->two_days_ago, NUM_MANUFACTURERS);

  printf("Printing yesterday\n");
  print_int_array(profit_history->yesterday, NUM_MANUFACTURERS);

  host_price_response(0,0);
  
  





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
