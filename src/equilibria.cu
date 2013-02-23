typedef struct
{
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
