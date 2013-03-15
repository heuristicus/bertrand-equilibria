#include "cutil_inline.h"
#include "curand.h"
#include "curand_kernel.h"
#include "math.h"
#include "time.h"
#include "sys/timeb.h"
#include "stdlib.h"
#include "limits.h"
#include "stdio.h"

// Whether to print lots about the current values to stdout
#define VERBOSE 1

// Whether the consumers choose which product to buy based on loyalty.
// Otherwise, they just pick the cheapest
#define LOYALTY_ENABLED 1

#define COMPUTE_ON_DEVICE 50
#define COMPUTE_ON_HOST 51
//#define PRICE_RESPONSE_COMPUTE COMPUTE_ON_HOST
#define PRICE_RESPONSE_COMPUTE COMPUTE_ON_DEVICE

//#define MODIFY_PRICE_COMPUTE COMPUTE_ON_HOST
#define MODIFY_PRICE_COMPUTE COMPUTE_ON_DEVICE

//#define UPDATE_LOYALTIES_COMPUTE COMPUTE_ON_HOST
#define UPDATE_LOYALTIES_COMPUTE COMPUTE_ON_DEVICE

#define CONSUMER_CHOICE_COMPUTE COMPUTE_ON_HOST
//#define CONSUMER_CHOICE_COMPUTE COMPUTE_ON_DEVICE

// Direction of price for the given manufacturer.
// Up means prices are increasing, down is decreasing
#define NUM_STRATEGIES 2
#define STRATEGY_UP 0
#define STRATEGY_DOWN 1

#define NUM_MANUFACTURERS 2
#define NUM_CONSUMERS 1000
#define BASE_MARGINAL 100
#define MAX_MARGINAL BASE_MARGINAL*3
#define PRICE_INCREMENT 2
// The price of any product cannot exceed this value multiplied by the marginal
// cost for that product.
#define MAX_PRICE_MULTIPLIER 5.0f

// The gradient/decay rate of the function used to determine
// fitness for roulette-wheel selection, which is used to
// find the manufacturer to buy from
#define STINGINESS_ALPHA 8.0f

// By how much we multiply the score of the preferred manufacturer
#define LOYALTY_MULTIPLIER 2.0f

// What additional price over the cheapest we are willing to consider.
// E.g. 0.5 means we never buy products 50% more expensive than cheapest
#define RIPOFF_MULTIPLIER 1.0f

// Define the number of blocks and shared memory sizes for the device functions
#define LOYALTY_NBLOCKS 100
#define LOYALTY_THREADS_PER_BLOCK NUM_CONSUMERS/LOYALTY_NBLOCKS // This needs to change if the number of customers changes
#define LOYALTY_SHAREDSIZE LOYALTY_THREADS_PER_BLOCK*NUM_MANUFACTURERS

#define NUM_PRODUCTS 1

#define PRICE_RESPONSE_SHAREDSIZE NUM_MANUFACTURERS

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
void print_2d_1d_int_array(int* data_in, unsigned int size1, unsigned int size2);
void print_int_array(int* data_in, unsigned int size);
int* calculate_num_purchases(int* purchases, unsigned int num_consumers);
void print_profit_struct(profits* profit, unsigned int num_manufacturers);
int get_max_ind(int* array, unsigned int size);
int get_min_ind(int* array, unsigned int size);
void put_plot_line(FILE* fp, int* arr, unsigned int size, int x);
void modify_price(int* marginal_cost, int* max_cost, int manufacturer_id, int product_id, int strategy, int* price_arr, int num_manufacturers);
int* manufacturer_loyalty_counts(int* loyal_arr, int num_manufacturers, int num_consumers);
__device__ int d_get_max_ind(int* array, unsigned int size);
__global__ void d_update_loyalties(int* choices, int* loyalties, unsigned int num_manufacturers,
                                   unsigned int num_customers);
void launch_update_loyalties(int* choices, int* loyalties, unsigned int num_consumers,
                             unsigned int num_manufacturers);

// Values for timing device functions
int loyalty_update_count;
float loyalty_update_total_millis;

int price_response_count;
float price_response_total_millis;

// dim1 = first dimension, dim2 is second
// So to do arr[1][5] -> idx(1, 5, width)
int idx(unsigned int dim1, unsigned int dim2, unsigned int width)
{
  if (dim2 >= width) 
  {
    fprintf(stderr, "Error! IndexOutOfBounds. dim2=%d, width=%d. Exiting...\n", 
            dim2, width);
    exit(-1);
  }
  
  return dim2 + dim1*width;
}

int val(int* array, unsigned int dim1, unsigned int dim2, unsigned int width) 
{
  return array[idx(dim1,dim2,width)];
}

void set_val(int* array, unsigned int dim1, unsigned int dim2, unsigned int width, int val) 
{
  array[idx(dim1,dim2,width)] = val;
}

// Please forgive me.
// No way to throw exceptions, and setting error codes is too complex.
// This crashes the kernel.
__device__ void die() 
{
  int* balls = (int*)0xffffffff;
  *balls = 5;
}

// dim1 = first dimension, dim2 is second
// So to do arr[1][5] -> idx(1, 5, width)
__device__ int d_idx(unsigned int dim1, unsigned int dim2, unsigned int width)
{
  if (dim2 >= width) 
  {
    //fprintf(stderr, "Error! IndexOutOfBounds. dim2=%d, width=%d. Exiting...\n", 
    //        dim2, width);
    die(); // Equivalent of throwing an exception since we can't
  }
  
  return dim2 + dim1*width;
}

__device__ int d_val(int* array, unsigned int dim1, unsigned int dim2, unsigned int width) 
{
  return array[d_idx(dim1,dim2,width)];
}

__device__ void d_set_val(int* array, unsigned int dim1, unsigned int dim2, unsigned int width, int val) 
{
  array[d_idx(dim1,dim2,width)] = val;
}

__device__ int d_get_min_ind(int* array, unsigned int size)
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

__device__ int d_get_cheapest_man(int* price, int product_id)
{
    int* price_arr_point = &price[product_id*NUM_MANUFACTURERS];
    return d_get_min_ind(price_arr_point, NUM_MANUFACTURERS);
}


// Each manufacturer has a strategy at a given moment in time.
// Either they are raising their profits or decreasing them. Here, we initialise
// these values to random strategies for the first time step
int* init_strategy() 
{
  int* price_strategy = (int*) malloc(NUM_MANUFACTURERS*sizeof(int));
  
  int i;
  for (i = 0; i < NUM_MANUFACTURERS; i++) 
  {
    // Randomly choose int between 0 and num of strategies-1
    float randVal = (float)rand()/RAND_MAX;
    price_strategy[i] = (int)(randVal*NUM_STRATEGIES);
  }
  return price_strategy;
}

// Initialises the marginal and maximum costs for each product. The maximum price is
// some multiple of the marginal cost.
int* init_marginal_cost()
{
    int* marginal_cost = (int*) malloc(NUM_PRODUCTS * sizeof(int));

    int i;
    
    for (i = 0; i < NUM_PRODUCTS; ++i) {
        float rval = (float)rand()/RAND_MAX;
        //marginal_cost[i] = (int)(rval * MAX_MARGINAL);
        marginal_cost[i] = BASE_MARGINAL+(i*(BASE_MARGINAL/10));
    
//    printf("Marginal cost for %s is %d.\n", products[i], marginal_cost[i]);
    }
    return marginal_cost;
}

// Initialises the maximum costs of products based on their marginal cost
int* init_max_cost(int* marginal_cost){
    int* max_cost = (int*) malloc(NUM_PRODUCTS * sizeof(int));

    int i;

    for (i = 0; i < NUM_PRODUCTS; ++i) {
        max_cost[i] = MAX_PRICE_MULTIPLIER * marginal_cost[i];
    }
    return max_cost; 
}

// Rand*MC*3 (roughly)
// Initialises the prices for each product.
// Price array: first dimension product, second dimension manufacturer 
// (flattened to 1d)
int* init_prices(int* marginal_cost)
{
  int i;
  int j;

  int* price = (int*) malloc(NUM_PRODUCTS * NUM_MANUFACTURERS * sizeof(int));
  const int width = NUM_MANUFACTURERS;

  for (i = 0; i < NUM_PRODUCTS; ++i) {
    for (j = 0; j < NUM_MANUFACTURERS; ++j) {
      float rval = (float)rand()/RAND_MAX;
      float val = marginal_cost[i] + (rval * marginal_cost[i]);
      set_val(price, i, j, width, val);
//      printf("Price for %s (%d) from manufacturer %d: %d\n", products[i], i, j, price[i][j]);
    }
  }
  return price;
}

// Uniformly distributed
int* init_loyalty()
{
    int* loyalty = (int*) malloc(NUM_CONSUMERS * sizeof(int));
    
    int i;
    int* counts = (int*) malloc(NUM_MANUFACTURERS * sizeof(int));
    
    for (i = 0; i < NUM_CONSUMERS; ++i) {
        loyalty[i] = select_loyalty();
        //	printf("Customer %d loyal to manufacturer %d\n", i, loyalty[i]);
        counts[loyalty[i]]++;
    }

    /* for (i = 0; i < NUM_MANUFACTURERS; ++i) { */
    /*     printf("Manufacturer %d has %d loyal customers.\n", i, counts[i]); */
    /* } */
    return loyalty;
}

// Returns uniform random number in the range [0, NUM_MANUFACTURERS]
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

// Initialise last two days of profits with fake values.
// All profits two days ago are set to 0 and for yesterday 
// are set to 1. Thus, all profits increase so currently
// active strategies are kept in place and acted on
profits* init_profits() 
{
    profits* profit_history = (profits*) malloc(sizeof(profits));
    profit_history->two_days_ago = (int*) malloc(sizeof(int)*NUM_MANUFACTURERS);
    profit_history->yesterday = (int*) malloc(sizeof(int)*NUM_MANUFACTURERS);
    profit_history->today = (int*) malloc(sizeof(int)*NUM_MANUFACTURERS);

    int man;
    for (man = 0; man < NUM_MANUFACTURERS; ++man)
    {
        profit_history->two_days_ago[man] = 0;
        profit_history->yesterday[man] = 1;
    }
    return profit_history;
}

// Computes the choice made for each product by each consumer. Puts the values for each consumer into the chosen_manufacturers array,
// which is assumed to be initialised with a size of num_consumers*num_products.
__global__ void device_consumer_choice(
  int* chosen_manufacturers, int* loyalty, int* price, 
  unsigned int num_manufacturers, unsigned int num_consumers, 
  unsigned int num_products, curandState* states, int seed){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int cons_id = tid; // Each thread deals with a single consumer
    curand_init(seed * tid, tid, 0, &states[tid]);
    for (int product_id = 0; product_id < num_products; product_id++){
        int cheapest_man = d_get_cheapest_man(price, product_id);
        if (cheapest_man == loyalty[cons_id]){
            chosen_manufacturers[cons_id + product_id * num_consumers] = cheapest_man;
        } else {
            float cheapest_price = (float) d_val(price, product_id, cheapest_man, num_manufacturers);
            float scores[NUM_PRODUCTS];

            float total_score = 0.0f;
    
            for (int man = 0; man < num_manufacturers; man++)
            {
                // equiv. of x in function
                int price_diff = d_val(price, product_id, man, num_manufacturers) - cheapest_price;
                float score;
                if (price_diff > RIPOFF_MULTIPLIER*cheapest_price) 
                {
                    score = 0;
                }
                else
                {
                    score = cheapest_price/(STINGINESS_ALPHA*price_diff + cheapest_price);
                    total_score += score;
                }

                if (man == loyalty[cons_id])
                {
                    score *= LOYALTY_MULTIPLIER;
                }

                scores[man] = score;
            }

            float ran = curand_uniform(&states[tid]) * total_score;

            float score_so_far = 0.0f;

            for (int man = 0; man < num_manufacturers; man++) 
            {
                score_so_far += scores[man];
                if (score_so_far >= ran)
                {
                    chosen_manufacturers[cons_id + product_id * num_consumers] = man;
                    break;
                }
            }
            
        }
    }
}

// Launch a kernel to compute the manufacturer that each customer chooses for each product.
void launch_consumer_choice(int* chosen_manufacturers, int* loyalty, int* price,
                            unsigned int num_manufacturers, unsigned int num_consumers,
                            unsigned int num_products){
    int blocks = 1;
    int threadsPerBlock = num_consumers;
  
    int* dev_loyalty;
    int* dev_chosen_manufacturers;
    int* dev_price;
    int loyalty_memsize = num_consumers * sizeof(int);
    int choose_memsize = num_consumers * num_products * sizeof(int);
    int price_memsize = num_products * num_manufacturers * sizeof(int);

    curandState* dev_states;
    
    cutilSafeCall(cudaMalloc((void**) &dev_states, threadsPerBlock*blocks*sizeof(curandState)));
    cutilSafeCall(cudaMalloc((void**) &dev_loyalty, loyalty_memsize));
    cutilSafeCall(cudaMalloc((void**) &dev_chosen_manufacturers, choose_memsize));
    cutilSafeCall(cudaMalloc((void**) &dev_price, price_memsize));

    cutilSafeCall(cudaMemcpy(dev_loyalty, loyalty, loyalty_memsize, cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(dev_chosen_manufacturers, chosen_manufacturers, choose_memsize,
                             cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(dev_price, price, price_memsize, cudaMemcpyHostToDevice));

    device_consumer_choice<<<blocks, threadsPerBlock>>>(dev_chosen_manufacturers, dev_loyalty,
                                                        dev_price, num_manufacturers, num_consumers,
                                                        num_products, dev_states, clock());

    cutilSafeCall(cudaMemcpy(chosen_manufacturers, dev_chosen_manufacturers, choose_memsize,
                             cudaMemcpyDeviceToHost));
    
    cutilSafeCall(cudaFree(dev_loyalty));
    cutilSafeCall(cudaFree(dev_price));
    cutilSafeCall(cudaFree(dev_chosen_manufacturers));
}


// Get the manufacturer ID fom which the consumer chooses to purchase the given product
int host_consumer_choice(int* loyalty, int* price, int consumer_id, int product_id,
                         int cheapest_man, int loyalty_enabled, int num_manufacturers) {
  if (! loyalty_enabled) 
  {
    return cheapest_man;
  }
  
  // If cheapest manufacturer is already preferred, pick that
  if (loyalty[consumer_id] == cheapest_man) 
  {
    //   printf("Preferred is cheapest. Returning %d\n",cheapest_man);
    return cheapest_man;
  }
  else
  {
    float cheapest_price = (float) val(price, product_id, cheapest_man, num_manufacturers);
    float* scores = (float*) malloc(sizeof(float)*num_manufacturers);

    float total_score = 0.0f;
    
    for (int man = 0; man < num_manufacturers; man++) 
    {
      // equiv. of x in function
      int price_diff = val(price, product_id, man, num_manufacturers) - cheapest_price;
      float score;
      if (price_diff > RIPOFF_MULTIPLIER*cheapest_price) 
      {
        score = 0;
      }
      else
      {
        score = cheapest_price/(STINGINESS_ALPHA*price_diff + cheapest_price);
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

    for (int man = 0; man < num_manufacturers; man++) 
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

// Get the strategy for the manufacturer based on previous profits
void host_price_response(profits* profit_history, int manufacturer_id, 
                         int* price_strategy_arr) {
  int current_strategy = price_strategy_arr[manufacturer_id];
  int profit1 = profit_history->two_days_ago[manufacturer_id];
  int profit2 = profit_history->yesterday[manufacturer_id];

  // If profit decreased, switch strategy
  if (profit1 > profit2) 
  {
    if (current_strategy == STRATEGY_UP) 
    {
      price_strategy_arr[manufacturer_id] = STRATEGY_DOWN;
    }
    else
    {
      price_strategy_arr[manufacturer_id] = STRATEGY_UP;
    }
  }
  else if (profit1 == profit2) {
    price_strategy_arr[manufacturer_id] = STRATEGY_DOWN;
  }
}

// Modifies the price the manufacturer charges for the given product based on
// the current strategy. The price can never exceed some multiple of the marginal
// cost, and can never fall below the marginal cost.
void modify_price(int* marginal_cost, int* max_cost, 
                  int manufacturer_id, int product_id, 
                  int strategy, int* price_arr, int num_manufacturers)
{
  int price_of_prod = val(price_arr, product_id, manufacturer_id, num_manufacturers);
  
  if (strategy == STRATEGY_UP && price_of_prod <= max_cost[product_id] - PRICE_INCREMENT) 
  {
    int new_price = price_of_prod + PRICE_INCREMENT;
    set_val(price_arr, product_id, manufacturer_id, num_manufacturers, new_price);
  }
  else if (strategy == STRATEGY_DOWN && price_of_prod >= marginal_cost[product_id] + PRICE_INCREMENT) 
  {
    int new_price = price_of_prod - PRICE_INCREMENT;
    set_val(price_arr, product_id, manufacturer_id, num_manufacturers, new_price);
  }
}

// Gets the ID of the manufacturer which has the cheapest product for the given ID.
int get_cheapest_man(int* price, int product_id)
{
  int* price_arr_point = &price[product_id*NUM_MANUFACTURERS];
  return get_min_ind(price_arr_point, NUM_MANUFACTURERS);
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

/* // The counts array will be modified to contain the total purchases for each manufacturer. It is */
/* // assumed to be initialised with the correct size. */
/* __global__ void calculate_num_purchases(int* counts, int* purchases, unsigned int num_consumers, */
/*                              unsigned int num_manufacturers){ */
    
/* } */

// Adds the profit for the given product into the profit_today array. The array
// should be initialised with zeroes on the first call
void profit_for_product(int* purchases, int* profit_today, int* price,
                        int marginal_cost, unsigned int num_manufacturers){
  for (int man_id = 0; man_id < num_manufacturers; man_id++){
    profit_today[man_id] += purchases[man_id] * (price[man_id] - marginal_cost);
  }
}

/* __global__ void profit_for_product(int* purchases, int* profit_today, int* price, int marginal_cost, unsigned int num_manufacturers){ */
    
/* } */

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
void update_loyalties(int* choices, int* loyalties, unsigned int num_consumers,
                      unsigned int num_manufacturers)
{
    for (int cons_id = 0; cons_id < num_consumers; cons_id++)
    {
        // Most purchased-from manufacturer
        int* choices_subarr = &choices[cons_id*num_manufacturers];
        loyalties[cons_id] = get_max_ind(choices_subarr, num_manufacturers);
    }
}

// Updates the loyalties of each customer after the purchases for the day have been made.
// The number of threads should be the number of consumers.
__global__ void d_update_loyalties(int* choices, int* loyalties, 
                                   unsigned int num_manufacturers,
                                   unsigned int num_consumers){
    int cons_id = threadIdx.x + blockDim.x*blockIdx.x;

    // Most purchased-from manufacturer
    int* choices_subarr = &choices[cons_id*num_manufacturers];
    loyalties[cons_id] = d_get_max_ind(choices_subarr, num_manufacturers);
}

// Updates the loyalties of each customer after the purchases for the day have been made.
// The number of threads should be the number of consumers. Loads data from global memory
// into shared in an attempt to speed up computation
__global__ void d_update_loyalties_shmem(int* choices, int* loyalties, 
                                   unsigned int num_manufacturers,
                                   unsigned int num_consumers)
{
    int cons_id = threadIdx.x + blockDim.x*blockIdx.x;
    int tid = threadIdx.x;

    __shared__ int c_shared[LOYALTY_SHAREDSIZE];
    int global_ref = cons_id*num_manufacturers;
    int shared_ref = tid*num_manufacturers;

    for (int man_id = 0; man_id < num_manufacturers; man_id++){
        c_shared[shared_ref + man_id] = choices[global_ref + man_id];
    }
    
    // Pointer to the start of the array for this consumer.
    int* choices_subarr = &c_shared[shared_ref];
    // Get the most purchased-from manufacturer and set loyalty to it.
    loyalties[cons_id] = d_get_max_ind(choices_subarr, num_manufacturers);
}

// Updates the loyalties of each customer after the purchases for the day have been made.
// The number of threads should be the number of consumers. Attempts to speed up computation
// using memory coalescing.
__global__ void d_update_loyalties_shmem_coop(int* choices, int* loyalties, 
                                   unsigned int num_manufacturers,
                                   unsigned int num_consumers)
{
    int cons_id = threadIdx.x + blockDim.x*blockIdx.x;
    int tid = threadIdx.x;
    // This is the number of array elements that should be loaded at each stage of the cooperative loading.
//    int shBlockSize = num_consumers/blockDim.x;

    __shared__ int c_shared[LOYALTY_SHAREDSIZE];
    int shared_ref = tid*num_manufacturers;
    
    for (int man_id = 0; man_id < num_manufacturers; man_id++){
        c_shared[blockDim.x * man_id + tid] = choices[num_consumers * man_id + cons_id];
//        loyalties[cons_id] = c_shared[num_consumers * man_id + tid];
//        loyalties[cons_id] = shBlockSize * man_id + tid;
    }

    __syncthreads();
    // Pointer to the start of the array for this consumer.
    int* choices_subarr = &c_shared[shared_ref];
    // Get the most purchased-from manufacturer and set loyalty to it.
    loyalties[cons_id] = d_get_max_ind(choices_subarr, num_manufacturers);
}



// Performs the necessary memory allocations and conversions and launches the
// kernel function to compute the updated loyalties.
void launch_update_loyalties(int* choices, int* loyalties, 
                             unsigned int num_consumers,
                             unsigned int num_manufacturers)
{
    int nblocks = LOYALTY_NBLOCKS, nthreads = LOYALTY_THREADS_PER_BLOCK;
    int choice_memsize = num_consumers * num_manufacturers * sizeof(int);
    int loyalty_memsize = num_consumers * sizeof(int);
    int* dev_choices;
    int* dev_loyalty;
    
    // Allocate device memory for both arrays
    cutilSafeCall(cudaMalloc((void**) &dev_choices, choice_memsize));
    cutilSafeCall(cudaMalloc((void**) &dev_loyalty, loyalty_memsize));

    // Copy the data into the device arrays. Only need to do this for the choices, since that
    // is the only data which is read - the device will overwrite values in the loyalties array.
    cutilSafeCall(cudaMemcpy(dev_choices, choices, choice_memsize, cudaMemcpyHostToDevice));

    unsigned int timer = 0;
    cutilCheckError(cutCreateTimer(&timer));
    cutilCheckError(cutStartTimer(timer));  

//    d_update_loyalties<<<nblocks, nthreads>>>(dev_choices, dev_loyalty, num_manufacturers, num_consumers);
//    d_update_loyalties_shmem<<<nblocks, nthreads>>>(dev_choices, dev_loyalty, num_manufacturers, num_consumers);
    d_update_loyalties_shmem_coop<<<nblocks, nthreads>>>(dev_choices, dev_loyalty, num_manufacturers, num_consumers);

    cutilCheckError(cutStopTimer(timer));
    loyalty_update_total_millis += cutGetTimerValue(timer);
    loyalty_update_count++;

    cutilSafeCall(cudaMemcpy(loyalties, dev_loyalty, loyalty_memsize, cudaMemcpyDeviceToHost));

    cutilSafeCall(cudaFree(dev_loyalty));
    cutilSafeCall(cudaFree(dev_choices));
}



// Get the index of the maximum value in the given array.
__device__ int d_get_max_ind(int* array, unsigned int size)
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

// Get tomorrow's strategy for each manufacturer
// This sets strategy only. Price needs to be set separately.
// Number of threads should be num of manufacturers
__global__ void device_price_response(int* price_strategy,
                                      int* profit_two_days_ago, 
                                      int* profit_yesterday) {
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  const int manufacturer_id = tid;

  int current_strategy = price_strategy[manufacturer_id];
  int profit1 = profit_two_days_ago[manufacturer_id];
  int profit2 = profit_yesterday[manufacturer_id];

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
}

// Get tomorrow's strategy for each manufacturer
// This sets strategy only. Price needs to be set separately.
// Number of threads should be num of manufacturers
__global__ void device_price_response_shmem(int* price_strategy,
                                      int* profit_two_days_ago, 
                                      int* profit_yesterday) {
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  const int man_id = tid;

  __shared__ int current_strategy[PRICE_RESPONSE_SHAREDSIZE];
  __shared__ int profit1[PRICE_RESPONSE_SHAREDSIZE];
  __shared__ int profit2[PRICE_RESPONSE_SHAREDSIZE];

  current_strategy[man_id] = price_strategy[man_id];
  profit1[man_id] = profit_two_days_ago[man_id];
  profit2[man_id] = profit_yesterday[man_id];
  
  // If profit decreased, switch strategy
  if (profit1[man_id] > profit2[man_id])
  {
    if (current_strategy[man_id] == STRATEGY_UP)
    {
      price_strategy[man_id] = STRATEGY_DOWN;
    }
    else
    {
      price_strategy[man_id] = STRATEGY_UP;
    }
  }
  else if (profit1[man_id] == profit2[man_id]) {
    price_strategy[man_id] = STRATEGY_DOWN;
  }
}

void launch_device_price_response(int* price_strategy,
                                  int* profit_two_days_ago, 
                                  int* profit_yesterday,
                                  int num_manufacturers)
{
  int blocks = 1;
  int threadsPerBlock = num_manufacturers;
  
  int* dev_price_strategy;
  int* dev_profit_two_days_ago;
  int* dev_profit_yesterday;
  int mem_size = num_manufacturers * sizeof(int);

  cutilSafeCall(cudaMalloc((void**) &dev_price_strategy, mem_size));
  cutilSafeCall(cudaMalloc((void**) &dev_profit_two_days_ago, mem_size));
  cutilSafeCall(cudaMalloc((void**) &dev_profit_yesterday, mem_size));

  cutilSafeCall(cudaMemcpy(dev_price_strategy, price_strategy, mem_size,
                           cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dev_profit_two_days_ago, profit_two_days_ago,
                           mem_size, cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dev_profit_yesterday, profit_yesterday,
                           mem_size, cudaMemcpyHostToDevice));


  unsigned int timer = 0;
  cutilCheckError(cutCreateTimer(&timer));
  cutilCheckError(cutStartTimer(timer));  

  /* device_price_response<<<blocks, threadsPerBlock>>>(dev_price_strategy, */
  /*                                                    dev_profit_two_days_ago, */
  /*                                                    dev_profit_yesterday); */

  device_price_response_shmem<<<blocks, threadsPerBlock>>>(dev_price_strategy,
                                                     dev_profit_two_days_ago,
                                                     dev_profit_yesterday);

  cutilCheckError(cutStopTimer(timer));
  price_response_total_millis += cutGetTimerValue(timer);
  price_response_count++;

  cutilSafeCall(cudaMemcpy(price_strategy, dev_price_strategy, mem_size,
                           cudaMemcpyDeviceToHost));
  cutilSafeCall(cudaMemcpy(profit_two_days_ago, dev_profit_two_days_ago,
                           mem_size, cudaMemcpyDeviceToHost));
  cutilSafeCall(cudaMemcpy(profit_yesterday, dev_profit_yesterday,
                           mem_size, cudaMemcpyDeviceToHost));

  cutilSafeCall(cudaFree(dev_price_strategy));
  cutilSafeCall(cudaFree(dev_profit_two_days_ago));
  cutilSafeCall(cudaFree(dev_profit_yesterday));
}


// Modifies the price the manufacturer charges for each product based on
// the current strategy. The price can never exceed some multiple of the marginal
// cost, and can never fall below the marginal cost.
// Number of threads should be num_manufacturers*num_products
__global__ void device_modify_price(int* strategy_arr, 
                                    int* price_arr, 
                                    int* max_cost_arr,
                                    int* marginal_cost_arr,
                                    int num_manufacturers,
                                    int num_products)
{
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  const int manufacturer_id = tid / num_products;
  const int product_id = tid % num_products;
  
  int price_of_prod = d_val(price_arr, product_id, manufacturer_id, num_manufacturers);
  
  if (strategy_arr[manufacturer_id] == STRATEGY_UP && 
      price_of_prod <= max_cost_arr[product_id] - PRICE_INCREMENT) 
  {
    int new_price = price_of_prod + PRICE_INCREMENT;
    d_set_val(price_arr, product_id, manufacturer_id, num_manufacturers, new_price);
  }
  else if (strategy_arr[manufacturer_id] == STRATEGY_DOWN && 
           price_of_prod >= marginal_cost_arr[product_id] + PRICE_INCREMENT) 
  {
    int new_price = price_of_prod - PRICE_INCREMENT;
    d_set_val(price_arr, product_id, manufacturer_id, num_manufacturers, new_price);
  }
}

void launch_device_modify_price(int* strategy_arr, 
                                int* price_arr, 
                                int* max_cost_arr,
                                int* marginal_cost_arr,
                                int num_manufacturers,
                                int num_products)
{
  int blocks = 1;
  int threadsPerBlock = num_manufacturers*num_products;
  
  // Mem size for arrays containing elements up to num_manufacturers 
  int man_mem_size = num_manufacturers * sizeof(int);

  // Mem size for arrays containing elements up to num_products
  int prod_mem_size = num_products * sizeof(int);

  // Mem size for arrays containing elements of num_products*num_manufacturers
  int man_prod_mem_size = num_manufacturers * num_products * sizeof(int);

  int* dev_strategy_arr;
  int* dev_price_arr;
  int* dev_max_cost_arr;
  int* dev_marginal_cost_arr;
  
  cutilSafeCall(cudaMalloc((void**) &dev_strategy_arr, man_mem_size));
  cutilSafeCall(cudaMalloc((void**) &dev_price_arr, man_prod_mem_size));
  cutilSafeCall(cudaMalloc((void**) &dev_max_cost_arr, prod_mem_size));
  cutilSafeCall(cudaMalloc((void**) &dev_marginal_cost_arr, prod_mem_size));

  cutilSafeCall(cudaMemcpy(dev_strategy_arr, strategy_arr, man_mem_size,
                           cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dev_price_arr, price_arr, man_prod_mem_size,
                           cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dev_max_cost_arr, max_cost_arr, prod_mem_size,
                           cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dev_marginal_cost_arr, marginal_cost_arr, prod_mem_size,
                           cudaMemcpyHostToDevice));

  device_modify_price<<<blocks, threadsPerBlock>>>(dev_strategy_arr,
                                                   dev_price_arr, 
                                                   dev_max_cost_arr,
                                                   dev_marginal_cost_arr,
                                                   num_manufacturers,
                                                   num_products);

  cutilSafeCall(cudaMemcpy(strategy_arr, dev_strategy_arr, man_mem_size,
                           cudaMemcpyDeviceToHost));
  cutilSafeCall(cudaMemcpy(price_arr, dev_price_arr, man_prod_mem_size,
                           cudaMemcpyDeviceToHost));
  cutilSafeCall(cudaMemcpy(max_cost_arr, dev_max_cost_arr, prod_mem_size,
                           cudaMemcpyDeviceToHost));
  cutilSafeCall(cudaMemcpy(marginal_cost_arr, dev_marginal_cost_arr,
                           prod_mem_size, cudaMemcpyDeviceToHost));

  cutilSafeCall(cudaFree(dev_strategy_arr));
  cutilSafeCall(cudaFree(dev_price_arr));
  cutilSafeCall(cudaFree(dev_max_cost_arr));
  cutilSafeCall(cudaFree(dev_marginal_cost_arr));
}

void print_timer_values() {
  if (UPDATE_LOYALTIES_COMPUTE == COMPUTE_ON_DEVICE)
  {
    float loyalty_avg_update_time = (float)loyalty_update_total_millis / (float)loyalty_update_count;
    printf("Total loyalty update time: %.10f ms\nAverage loyalty update time: %.10f ms\n\n", loyalty_update_total_millis, loyalty_avg_update_time);
  }
  
  if (PRICE_RESPONSE_COMPUTE == COMPUTE_ON_DEVICE)
  {
    float price_response_avg_time = (float)price_response_total_millis / (float)price_response_count;
    printf("Total price response time: %.10f ms\nAverage price response time: %.10f ms\n\n", price_response_total_millis, price_response_avg_time);
  }
}

void host_equilibriate(int* price, int* loyalty,
                       profits* profit, int* price_strategy,
                       int* marginal_cost, int* max_cost, 
                       int days, int loyalty_enabled, 
                       char* profitFilename, char* priceFilename,
                       char* loyaltyFilename, int seed)
{
  int day_num;
  int man_id, prod_id, cons_id;
  
  FILE* profitFile = fopen(profitFilename, "w");
  FILE* priceFile = fopen(priceFilename, "w");
  FILE* loyalFile = fopen(loyaltyFilename, "w");
  
  for (day_num = 0; day_num < days; day_num++)
  {
    if (VERBOSE)
    {
      printf("Old prices (line = product):\n");
      print_2d_1d_int_array(price, NUM_PRODUCTS, NUM_MANUFACTURERS);

      printf("Strategies (0 is up, 1 is down): \n");
      print_int_array(price_strategy, NUM_MANUFACTURERS);
    }
    
    if (PRICE_RESPONSE_COMPUTE == COMPUTE_ON_DEVICE) 
    {
      launch_device_price_response(price_strategy,
                                   profit->two_days_ago, 
                                   profit->yesterday,
                                   NUM_MANUFACTURERS);
      printf("Launch_device_price_response successful!\n");
      if (VERBOSE)
      {
        print_int_array(price_strategy, NUM_MANUFACTURERS);
      }
    }
    else 
    {
      for (man_id = 0; man_id < NUM_MANUFACTURERS; man_id++){
        host_price_response(profit, man_id, price_strategy);
      }
    }

    if (MODIFY_PRICE_COMPUTE == COMPUTE_ON_DEVICE)
    {
      launch_device_modify_price(price_strategy, 
                                 price,
                                 max_cost,
                                 marginal_cost,
                                 NUM_MANUFACTURERS,
                                 NUM_PRODUCTS);

      printf("Launch_device_modify_price successful!\n");
      if (VERBOSE)
      {
        print_2d_1d_int_array(price, NUM_PRODUCTS, NUM_MANUFACTURERS);
      }
    }
    else
    {
      for (man_id = 0; man_id < NUM_MANUFACTURERS; man_id++){
        for (prod_id = 0; prod_id < NUM_PRODUCTS; prod_id++){
          modify_price(marginal_cost, max_cost, man_id,
                       prod_id, price_strategy[man_id],
                       price, NUM_MANUFACTURERS);
        }
      }
    }
    
    if (VERBOSE) 
    {
      printf("New prices (line = product):\n");
      print_2d_1d_int_array(price, NUM_PRODUCTS, NUM_MANUFACTURERS);
    }

    // This array contains the number of picks that a consumer has made from
    // each manufacturer. The first dimension is the consumer id, and the second
    // is the manufacturer.
    int* cons_choices = (int*) calloc(NUM_CONSUMERS * NUM_MANUFACTURERS, sizeof(int));


    if (CONSUMER_CHOICE_COMPUTE == COMPUTE_ON_HOST)
    {
      int* picks = (int*)malloc(sizeof(int) * NUM_CONSUMERS);

      for (prod_id = 0; prod_id < NUM_PRODUCTS; prod_id++){
        int cheapest = get_cheapest_man(price, prod_id);
        // TODO: Calculate the scores for this product here, rather than multiple times
        // in the consumer choice function.
        for (cons_id = 0; cons_id < NUM_CONSUMERS; cons_id++){
          picks[cons_id] = host_consumer_choice(loyalty, price, cons_id, 
                                                prod_id, cheapest, loyalty_enabled,
                                                NUM_MANUFACTURERS);
          // Increment the number of times the consumer picked the manufacturer
          // returned from the host_consumer_choice function
          int new_val = val(cons_choices, cons_id, picks[cons_id], NUM_MANUFACTURERS) + 1;
          set_val(cons_choices, cons_id, picks[cons_id], NUM_MANUFACTURERS, new_val);
        }
        int* counts = calculate_num_purchases(picks, NUM_CONSUMERS, NUM_MANUFACTURERS);

        if (VERBOSE)
        {
          printf("Printing picks for each consumer.\n");
          print_int_array(picks, NUM_CONSUMERS);
          printf("Number of purchases for each product:\n");
          print_int_array(counts, NUM_MANUFACTURERS);

          printf("ProfitToday before prod %d: %d\n", prod_id, profit->today[0]);
        }
      
        int* price_arr_point = &price[prod_id*NUM_MANUFACTURERS];
      
        profit_for_product(counts, profit->today, price_arr_point, 
                           marginal_cost[prod_id], NUM_MANUFACTURERS);
        //       print_profit_struct(profit, NUM_MANUFACTURERS);
      }
    }
    else
    {
        int* picks_2d = (int*)calloc(NUM_CONSUMERS * NUM_PRODUCTS, sizeof(int));
      
        launch_consumer_choice(picks_2d, 
                               loyalty,
                               price, 
                               NUM_MANUFACTURERS,
                               NUM_CONSUMERS,
                               NUM_PRODUCTS);
      
        if (VERBOSE)
        {      
          printf("Printing scores from picks.\n");
          print_2d_1d_int_array(picks_2d, NUM_PRODUCTS, NUM_CONSUMERS);
        }
        
        for (prod_id = 0; prod_id < NUM_PRODUCTS; prod_id++)
        {
          int cheapest = get_cheapest_man(price, prod_id);
          // Get picks for this cons out of flattened 2D array
          int* picks = &picks_2d[prod_id*NUM_CONSUMERS];

          for (cons_id = 0; cons_id < NUM_CONSUMERS; cons_id++){
            // Increment the number of times the consumer picked the manufacturer
            // returned from the host_consumer_choice function
            int new_val = val(cons_choices, cons_id, picks[cons_id], NUM_MANUFACTURERS) + 1;
          set_val(cons_choices, cons_id, picks[cons_id], NUM_MANUFACTURERS, new_val);
        }

        int* counts = calculate_num_purchases(picks, NUM_CONSUMERS, NUM_MANUFACTURERS);

        if (VERBOSE)
        {      
          printf("Number of purchases for each product:\n");
          print_int_array(counts, NUM_MANUFACTURERS);

          printf("ProfitToday before prod %d: %d\n", prod_id, profit->today[0]);
        }
        
        int* price_arr_point = &price[prod_id*NUM_MANUFACTURERS];
      
        profit_for_product(counts, profit->today, price_arr_point, marginal_cost[prod_id], NUM_MANUFACTURERS);
        if (VERBOSE)
        {
          print_profit_struct(profit, NUM_MANUFACTURERS);
        }
      }
    }
    
    if (UPDATE_LOYALTIES_COMPUTE == COMPUTE_ON_HOST) 
    {
      update_loyalties(cons_choices, loyalty, NUM_CONSUMERS, NUM_MANUFACTURERS);
    }
    else
    {
      launch_update_loyalties(cons_choices, 
                              loyalty,
                              NUM_CONSUMERS,
                              NUM_MANUFACTURERS);
    }
    
    if (VERBOSE)
    {      
      printf("Loyalties:\n");
      print_int_array(loyalty, NUM_CONSUMERS);
    }
    
    put_plot_line(profitFile, profit->today, NUM_MANUFACTURERS, day_num);
    int prod_to_print = 0;
    int* price_arr_point = &price[prod_to_print*NUM_MANUFACTURERS];
    put_plot_line(priceFile, price_arr_point, NUM_MANUFACTURERS, day_num);
    int* ct = manufacturer_loyalty_counts(loyalty, NUM_MANUFACTURERS, NUM_CONSUMERS);
    put_plot_line(loyalFile, ct, NUM_MANUFACTURERS, day_num);
    // swap the pointers inside the profit struct so that we can overwrite without needing to free
    swap_profit_pointers(profit, NUM_MANUFACTURERS);

    printf("A new day dawns (%d).\n\n", day_num);
  }

  print_timer_values();

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
  fflush(fp);
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

// Print a 2d array represented as a 1d array in 2d format
void print_2d_1d_int_array(int* data_in, unsigned int size1, unsigned int size2){
  for (int i = 0; i < size1; i++)
  {
    printf("[");
    for (int j = 0; j < size2 - 1; j++) 
    {
      printf("%d,", val(data_in, i, j, size2));
    }
    printf("%d]\n", val(data_in, i, size2-1, size2));
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
  if (argc < 7) {
      printf("Too few arguments received.\n");
      printf("Usage: %s nthreads nblocks ndays profit_outfile price_outfile loyalty_outfile [seed]\n", argv[0]);
      exit(1);
  }

  int threadsPerBlock = atoi(argv[1]);
  int blocksPerGrid = atoi(argv[2]);
  int days = atoi(argv[3]);
  char* profitFilename = argv[4];
  char* priceFilename = argv[5];
  char* loyaltyFilename = argv[6];

  int devID;
  cudaDeviceProp props;

  // get number of SMs on this GPU
  cutilSafeCall(cudaGetDevice(&devID));
  cutilSafeCall(cudaGetDeviceProperties(&props, devID));

  // ---------------------------------------------

  // If more than 7 arguments received, there should be a seed present so use
  // the seed to initialise the random number generator. Otherwise, just use
  // the current time.
  int seed;
  if (argc > 7)
      seed = atoi(argv[7]);
  else 
      seed = time(NULL);
  srand(seed);

  // Start of hard work...
  clock_t start_time = clock();
  time_t start, end;
  time(&start);

  int* loyalty = init_loyalty();
  int* marginal_cost = init_marginal_cost();
  int* max_cost = init_max_cost(marginal_cost);
  int* price = init_prices(marginal_cost);
  int* price_strategy = init_strategy();
  profits* profit_history = init_profits();

  host_equilibriate(price, loyalty, profit_history, price_strategy, marginal_cost, max_cost, days, LOYALTY_ENABLED, profitFilename, priceFilename, loyaltyFilename, seed);

  clock_t end_time = clock();
  time(&end);
  
  printf("CPU time taken: %fms\n", (double)(end_time-start_time)/CLOCKS_PER_SEC);
  printf("Wall clock time taken: %d secs\n", (int)(end-start));
  
  

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
  cudaThreadExit(); 
}
