#include "math.h"
#include "time.h"
#include "stdlib.h"
#include "limits.h"
#include "stdio.h"

// Compile with gcc --std=c99 -lm -g -Wall -Werror -o equilibria equilibria.c

//#define BLOCK_SIZE 32
//#define GRID_SIZE 32
#define NUM_MANUFACTURERS 2
#define NUM_CONSUMERS 50
#define MAX_MARGINAL 250

const char* products[] = {"milk", "bread", "toilet_paper", "butter", "jam", "cheese"};

int select_loyalty();

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


void init_marginal()
{
    marginal_cost = malloc((sizeof(products)/sizeof(char*)) * sizeof(int));
    srand(time(NULL));

    int i;
    
    for (i = 0; i < sizeof(products)/sizeof(char*); ++i) {
	float rval = rand();
	printf("rval is %f\n", rval);
	marginal_cost[i] = (int)(rval/INT_MAX * MAX_MARGINAL);
	printf("Marginal cost for %s is %d.\n", products[i], marginal_cost[i]);
    }
    
}

// Rand*MC*3 (roughly)
void init_prices() { // Mr Michalewicz
    int i, j;

    price = malloc(sizeof(products)/sizeof(char*) * sizeof(int*));

    for (i = 0; i < sizeof(products)/sizeof(char*); ++i) {
	price[i] = malloc(NUM_MANUFACTURERS * sizeof(int));
	for (j = 0; j < NUM_MANUFACTURERS; ++j) {
	    price[i][j] = (rand()/INT_MAX) * marginal_cost[i] * 3;
	}
    }
}

// Uniformly distributed
void init_loyalty() { // Mr Michalewicz
    loyalty = malloc(NUM_CONSUMERS * sizeof(int));
    
    int i;
    
    for (i = 0; i < NUM_CONSUMERS; ++i) {
	loyalty[i] = select_loyalty();
    }
}

int select_loyalty()
{
    int i;

//    int rval = rand()/INT_MAX;
    
    for (i = 0; i < NUM_MANUFACTURERS; ++i) {
	
    }

    return 0;
}

// Gaussian over population
void init_income() { // Mr Michalewicz
}

// Get the manufacturer ID fom which the consumer chooses to 
// purchase the given product
int host_consumer_choice(int consumer_id, int product_id) { // Mr Caine
    return 0;
}

// Get tomorrow's price for the given product ID
int host_price_response(int manufacturer_id, int product_id) { // Mr Caine
    return 0;
}

void host_equilibriate(int** price, int** consumption, int* income, int* loyalty, profits* profit) {
}

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
    init_income();
    init_loyalty();
    init_marginal();
    init_prices();
}
