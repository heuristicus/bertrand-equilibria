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
#define BASE_INCOME 20000

const char* products[] = {"milk", "bread", "toilet_paper", "butter", "jam", "cheese"};

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


void init_marginal()
{
    marginal_cost = malloc((sizeof(products)/sizeof(char*)) * sizeof(int));

    int i;
    
    for (i = 0; i < sizeof(products)/sizeof(char*); ++i) {
	float rval = (float)rand()/RAND_MAX;
	marginal_cost[i] = (int)(rval * MAX_MARGINAL);
	printf("Marginal cost for %s is %d.\n", products[i], marginal_cost[i]);
    }
}

// Rand*MC*3 (roughly)
void init_prices()
{
    int i, j;

    price = malloc(sizeof(products)/sizeof(char*) * sizeof(int*));

    for (i = 0; i < sizeof(products)/sizeof(char*); ++i) {
	price[i] = malloc(NUM_MANUFACTURERS * sizeof(int));
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
    loyalty = malloc(NUM_CONSUMERS * sizeof(int));
    
    int i;

    int* counts = malloc(NUM_MANUFACTURERS * sizeof(int));
    
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
    income = malloc(NUM_CONSUMERS * sizeof(int));
    
    int i;
    for (i = 0; i < NUM_CONSUMERS; ++i) {
	income[i] = BASE_INCOME * (positive_gaussrand() + 1);
	printf("Income of household %d: %d\n", i, income[i]);
    }
}

/*
 * Generate a gaussian random value in the interval [0,infinity]
 */
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
    srand(time(NULL));

    int i;
    
    for (i = 0; i < 100; ++i) {
	printf("%lf\n", positive_gaussrand() + 1);
    }


    init_income();
    init_loyalty();
    init_marginal();
    init_prices();
}
