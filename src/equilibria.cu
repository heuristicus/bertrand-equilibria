#include "cutil_inline.h"

#include "math.h"
#include "time.h"
#include "stdlib.h"
#include "limits.h"
#include "stdio.h"

#define ITERS 10

// Direction of price for the given manufacturer.
// Up means prices are increasing, down is decreasing
#define NUM_STRATEGIES 2
#define STRATEGY_UP 0
#define STRATEGY_DOWN 1

//#define BLOCK_SIZE 32
//#define GRID_SIZE 32
#define NUM_MANUFACTURERS 2
#define NUM_CONSUMERS 100
#define MAX_MARGINAL 250
#define BASE_INCOME 20000
#define PRICE_INCREMENT 5
// The price of any product cannot exceed this value multiplied by the marginal
// cost for that product.
#define MAX_PRICE_MULTIPLIER 5.0f 

// The gradient/decay rate of the function used to determine
// fitness for roulette-wheel selection, which is used to
// find the manufacturer to buy from
#define LOYALTY_ALPHA 8.0f

// By how much we multiply the score of the preferred manufacturer
#define LOYALTY_MULTIPLIER 2.0f

// What additional price over the cheapest we are willing to consider.
// E.g. 0.5 means we never buy products 50% more expensive than cheapest
#define RIPOFF_MULTIPLIER 1.0f

const char* products[] = {"milk", "bread", "toilet_paper", "butter", "bacon", "cheese"};
int NUM_PRODUCTS = sizeof(products)/sizeof(char*);

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

// Arrays mapping manufacturer ID to profit on each day
typedef struct
{
  int* two_days_ago;
  int* yesterday;
  int* today;
} profits;

int select_loyalty();
double gaussrand();
double positive_gaussrand();
void print_array(float*, unsigned int);
void print_2d_array(float** data_in, unsigned int size1, unsigned int size2);
void print_2d_int_array(int** data_in, unsigned int size1, unsigned int size2);
void print_int_array(int* data_in, unsigned int size);
int* calculate_num_purchases(int* purchases, unsigned int num_consumers);
void print_profit_struct(profits* profit, unsigned int num_manufacturers);
int get_max_ind(int* array, unsigned int size);
int get_min_ind(int* array, unsigned int size);
void put_plot_line(FILE* fp, int* arr, unsigned int size, int x);
void modify_price(int manufacturer_id, int product_id, int strategy);
int* manufacturer_loyalty_counts(int* loyal_arr, int num_manufacturers, int num_consumers);

// First dimension is product ID
// Second dimension is manufacturer ID
// Price is in pence
int** price; // 2D array

int* marginal_cost;

int* max_cost;

// First dimension is product ID
// Second dimension is consumer ID
// Consumption is units of product consumed per day
int** consumption; // 2D array

// Array mapping consumer ID to income in pence per day
int* income;

// Array mapping consumer ID to manufacturer ID
// Showing which manufacturer this consumer currently prefers
int* loyalty; 

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

// Each manufacturer has a strategy at a given moment in time.
// Either they are raising their profits or decreasing them. Here, we initialise
// these values to random strategies for the first time step
void init_strategy() 
{
  price_strategy = (int*) malloc(NUM_MANUFACTURERS*sizeof(int));
  
  int i;
  for (i = 0; i < NUM_MANUFACTURERS; i++) 
  {
    // Randomly choose int between 0 and num of strategies-1
    float randVal = (float)rand()/RAND_MAX;
    price_strategy[i] = (int)(randVal*NUM_STRATEGIES);
  }
}


// Initialises the marginal and maximum costs for each product. The maximum price is
// some multiple of the marginal cost.
void init_marginal_and_max()
{
  marginal_cost = (int*) malloc(NUM_PRODUCTS * sizeof(int));
  max_cost = (int*) malloc(NUM_PRODUCTS * sizeof(int));

  int i;
    
  for (i = 0; i < NUM_PRODUCTS; ++i) {
    float rval = (float)rand()/RAND_MAX;
    marginal_cost[i] = (int)(rval * MAX_MARGINAL);
    printf("Marginal cost for %s is %d.\n", products[i], marginal_cost[i]);
    max_cost[i] = MAX_PRICE_MULTIPLIER * marginal_cost[i];
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
      printf("Price for %s (%d) from manufacturer %d: %d\n", products[i], i, j, price[i][j]);
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
int host_consumer_choice(int consumer_id, int product_id, int cheapest_man) {
  // If cheapest manufacturer is already preferred, pick that
  if (loyalty[consumer_id] == cheapest_man) 
  {
    //   printf("Preferred is cheapest. Returning %d\n",cheapest_man);
    return cheapest_man;
  }
  else
  {
    float cheapest_price = (float) price[product_id][cheapest_man];
    float* scores = (float*) malloc(sizeof(float)*NUM_MANUFACTURERS);

    float total_score = 0.0f;
    
    for (int man = 0; man < NUM_MANUFACTURERS; man++) 
    {
      // equiv. of x in function
      int price_diff = price[product_id][man] - cheapest_price;
      float score;
      if (price_diff > RIPOFF_MULTIPLIER*cheapest_price) 
      {
        score = 0;
      }
      else
      {
        score = cheapest_price/(LOYALTY_ALPHA*price_diff + cheapest_price);
        total_score += score;
      }

      if (man == loyalty[consumer_id])
      {
        score *= LOYALTY_MULTIPLIER;
      }

      scores[man] = score;
    }

    float ran = (float)rand() / RAND_MAX * total_score;
    float score_so_far = 0.0f;


    printf("Scores array: ");
    print_array(scores, NUM_MANUFACTURERS);
    printf("Rand is %.5f\n", ran);

    for (int man = 0; man < NUM_MANUFACTURERS; man++) 
    {
      score_so_far += scores[man];
      if (score_so_far >= ran)
      {
        return man;
      }
    }
  }
  
  // Should have returned by now, so return -1 to crash or segfault or something
  fprintf(stderr, "Error! Didn't select anything in roulette wheel selection inside "\
          "host_consumer_choice. Exiting...\n");
  exit(1);
  return -1;
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
  else if (profit1 == profit2) {
    price_strategy[manufacturer_id] = STRATEGY_DOWN;
  }

  modify_price(manufacturer_id, product_id, price_strategy[manufacturer_id]);
}

// Modifies the price the manufacturer charges for the given product based on
// the current strategy. The price can never exceed some multiple of the marginal
// cost, and can never fall below the marginal cost.
void modify_price(int manufacturer_id, int product_id, int strategy)
{

  if (strategy == STRATEGY_UP && price[product_id][manufacturer_id] <= max_cost[product_id] + PRICE_INCREMENT)
    price[product_id][manufacturer_id] += PRICE_INCREMENT;
  else if (strategy == STRATEGY_DOWN && price[product_id][manufacturer_id] >= marginal_cost[product_id] - PRICE_INCREMENT)
    price[product_id][manufacturer_id] -= PRICE_INCREMENT;
  
}

// Gets the ID of the manufacturer which has the cheapest product for the given ID.
int get_cheapest_man(int product_id)
{
  return get_min_ind(price[product_id], NUM_MANUFACTURERS);
}

// We pass in the array of integers containing which manufacturer each
// consumer chooses based on the host_consumer_choice function. The return
// is an array containing the number of purchases made for each manufacturer
int* calculate_num_purchases(int* purchases, unsigned int num_consumers,
                             unsigned int num_manufacturers){
  int* counts = (int*)calloc(num_manufacturers, sizeof(int));
  for (int consumer_num = 0; consumer_num < num_consumers; consumer_num++){
    counts[purchases[consumer_num]]++;
  }

  return counts;
}

// Adds the profit for the given product into the profit_today array. The array
// should be initialised with zeroes on the first call
void profit_for_product(int* purchases, int* profit_today, int* price,
                        int marginal_cost, unsigned int num_manufacturers){
  for (int man_id = 0; man_id < num_manufacturers; man_id++){
    profit_today[man_id] += purchases[man_id] * (price[man_id] - marginal_cost);
  }
}

// Shifts pointers around in the given profit struct so that today's profits are
// yesterdays, and yesterday's are the profits two days ago. The array used to
// store the profits two days ago is zeroed and set to be used by the today array
void swap_profit_pointers(profits* profit, unsigned int num_manufacturers)
{
  int* tmp = profit->two_days_ago;
  profit->two_days_ago = profit->yesterday;
  profit->yesterday = profit->today;
  profit->today = tmp;
  bzero(profit->today, sizeof(int) * num_manufacturers);
}

// Update the loyalties of customers based on the number of purchases
// made from each manufacturer during the last day
void update_loyalties(int** choices, int* loyalties, unsigned int num_consumers, unsigned int num_manufacturers)
{
  for (int cons_id = 0; cons_id < num_consumers; cons_id++)
  {
    // Most purchased-from manufacturer
    int chosen = get_max_ind(choices[cons_id], num_manufacturers);

    // If we have purchased equally, we do not switch allegiance
    if (choices[chosen] != choices[loyalties[cons_id]]) {
      loyalties[cons_id] = chosen;
    }
  }
}

void host_equilibriate(int** price, int** consumption, int* income, int* loyalty, profits* profit, int days, char* profitFilename, char* priceFilename)
{
  int day_num;
  int man_id, prod_id, cons_id;
  
  char* ll = "loyalty.txt";
  FILE* profitFile = fopen(profitFilename, "w");
  FILE* priceFile = fopen(priceFilename, "w");
  FILE* loyalFile = fopen(ll, "w");
  
  for (day_num = 0; day_num < days; day_num++)
  {
    printf("Old prices (line = product):\n");
    print_2d_int_array(price, NUM_PRODUCTS, NUM_MANUFACTURERS);

    printf("Strategies (0 is up, 1 is down): \n");
    print_int_array(price_strategy, NUM_MANUFACTURERS);
    
    for (man_id = 0; man_id < NUM_MANUFACTURERS; man_id++){
      for (prod_id = 0; prod_id < NUM_PRODUCTS; prod_id++){
        host_price_response(man_id, prod_id);
      }
    }

    printf("New prices (line = product):\n");
    print_2d_int_array(price, NUM_PRODUCTS, NUM_MANUFACTURERS);


    int* picks = (int*)malloc(sizeof(int) * NUM_CONSUMERS);
    // This array contains the number of picks that a consumer has made from
    // each manufacturer. The first dimension is the consumer id, and the second
    // is the manufacturer.
    int** cons_choices = (int**) malloc(NUM_CONSUMERS * sizeof(int*));
    for (int i = 0; i < NUM_CONSUMERS; i++){
      cons_choices[i] = (int*) calloc(sizeof(int), NUM_MANUFACTURERS);
    }

    for (prod_id = 0; prod_id < NUM_PRODUCTS; prod_id++){
      int cheapest = get_cheapest_man(prod_id);
      // TODO: Calculate the scores for this product here, rather than multiple times
      // in the consumer choice function.
      for (cons_id = 0; cons_id < NUM_CONSUMERS; cons_id++){
        picks[cons_id] = host_consumer_choice(cons_id, prod_id, cheapest);
        cons_choices[cons_id][picks[cons_id]]++;
      }
      int* counts = calculate_num_purchases(picks, NUM_CONSUMERS, NUM_MANUFACTURERS);
      printf("Number of purchases for each product:\n");
      print_int_array(counts, NUM_MANUFACTURERS);
      profit_for_product(counts, profit->today, price[prod_id], marginal_cost[prod_id], NUM_MANUFACTURERS);
      print_profit_struct(profit, NUM_MANUFACTURERS);
    }

    update_loyalties(cons_choices, loyalty, NUM_CONSUMERS, NUM_MANUFACTURERS);
    printf("Loyalties:\n");
    print_int_array(loyalty, NUM_CONSUMERS);
    printf("Printing cons choices.\n");
    print_2d_int_array(cons_choices, NUM_CONSUMERS, NUM_MANUFACTURERS);

    put_plot_line(profitFile, profit->today, NUM_MANUFACTURERS, day_num);
    int prod_to_print = 0;
    put_plot_line(priceFile, price[prod_to_print], NUM_MANUFACTURERS, day_num);
    int* ct = manufacturer_loyalty_counts(loyalty, NUM_MANUFACTURERS, NUM_CONSUMERS);
    put_plot_line(loyalFile, ct, NUM_MANUFACTURERS, day_num);
    // swap the pointers inside the profit struct so that we can overwrite without needing to free
    swap_profit_pointers(profit, NUM_MANUFACTURERS);

    printf("A new day dawns.\n\n\n\n\n\n");
  }

  fclose(profitFile);
  fclose(priceFile);
  fclose(loyalFile);
}

int* manufacturer_loyalty_counts(int* loyal_arr, int num_manufacturers, int num_consumers)
{
  int* counts = (int*)calloc(num_manufacturers, sizeof(int));
  for (int consumer_num = 0; consumer_num < num_consumers; consumer_num++){
    counts[loyal_arr[consumer_num]]++;
  }
  return counts;
}

// Writes the given array into the provided file pointer. The x value
// is printed before the values in the array.
void put_plot_line(FILE* fp, int* arr, unsigned int size, int x)
{
  fprintf(fp, "%d", x);

  for (int i = 0; i < size; i++)
  {
    fprintf(fp, " %d", arr[i]);
  }

  fprintf(fp, "\n");
}

int get_min_ind(int* array, unsigned int size)
{
  int best = 0;
  for (int i = 1; i < size; i++)
  {
    if (array[i] < array[best])
    {
      best = i;
    }
  }
  return best;
}

int get_max_ind(int* array, unsigned int size)
{
  int best = 0;
  for (int i = 1; i < size; i++)
  {
    if (array[i] > array[best])
    {
      best = i;
    }
  }
  return best;
}

void copy_array(float* from, float* to, unsigned int size) {
  for (int i = 0; i < size; i++) {
    to[i] = from[i];
  }
}

void print_array(float* data_in, unsigned int size)
{
  for (int i=0; i < size-1; i++) {
    printf("%f,", data_in[i]);
  }
  printf("%f\n", data_in[size-1]);
}

void print_profit_struct(profits* profit, unsigned int num_manufacturers)
{
  printf("Profits\nTwo days ago: ");
  print_int_array(profit->two_days_ago, num_manufacturers);
  printf("Yesterday ");
  print_int_array(profit->yesterday, num_manufacturers);
  printf("Today ");
  print_int_array(profit->today, num_manufacturers);
}

// Size1 is for the top level array, size2 for the lower.
void print_2d_array(float** data_in, unsigned int size1, unsigned int size2){
  for (int i = 0; i < size1; i++){
    print_array(data_in[i], size2);
  }
}

void print_2d_int_array(int** data_in, unsigned int size1, unsigned int size2){
  for (int i = 0; i < size1; i++){
    print_int_array(data_in[i], size2);
  }
}

void print_int_array(int* data_in, unsigned int size) {
  for (int i=0; i < size-1; i++) {
    printf("%d,", data_in[i]);
  }
  printf("%d\n", data_in[size-1]);
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
  if (argc != 6) {
    printf("Please input five arguments: threads, blocks, number of days and output filenames of profit and price files\n");
    exit(1);
  }
  int threadsPerBlock = atoi(argv[1]);
  int blocksPerGrid = atoi(argv[2]);
  int days = atoi(argv[3]);
  char* profitFilename = argv[4];
  char* priceFilename = argv[5];


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
  init_marginal_and_max();
  init_prices();

  init_strategy();
  init_profits();

  /* printf("Printing two days ago\n"); */
  /* print_int_array(profit_history->two_days_ago, NUM_MANUFACTURERS); */

  /* printf("Printing yesterday\n"); */
  /* print_int_array(profit_history->yesterday, NUM_MANUFACTURERS); */

  /* host_price_response(0,0); */
  
  /* int product_id = 0; */
  /* int cheapest_man = get_cheapest_man(product_id); */
  
  /* // Cons, product_id, cheapest   */
  /* for (int cons = 0; cons < NUM_CONSUMERS; cons++) { */
  /*   int bought_man = host_consumer_choice(cons, product_id, cheapest_man); */
  /*   printf("Cons=%d, Loyalty=%d, Bought_man=%d\n\n",cons, loyalty[cons], bought_man); */
  /* } */

  print_2d_int_array(price, NUM_PRODUCTS, NUM_MANUFACTURERS);
  printf("Loyalties:\n");
  print_int_array(loyalty, NUM_CONSUMERS);
  host_equilibriate(price, consumption, income, loyalty, profit_history, days, profitFilename, priceFilename);



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
