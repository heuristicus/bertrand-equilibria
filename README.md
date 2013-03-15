Bertrand Equilibria
===================

Parallel Programming project in CUDA simulating Bertrand equilibria in duopolies

Execution Instructions
===================
`run` script in `/src` compiles the code, runs it and plots the output data.

Usage: `src/run DAYS SEED`

The days parameter specifies how many days the simulation should be run for.
The seed parameter sets the random number generator seed.

NOTE: The run script runs the simulation in the background, so in some cases
a large number (1000+) of days may mean that the simulation cannot be terminated
until it completes.

Important parameters which can be tweaked can be found at the top of the equilibria.cu file

- `NUM_CONSUMERS` - The number of consumers.
- `NUM_MANUFACTURERS` - The number of manufacturers.
- `STINGINESS_ALPHA` - Extent to which deviation from the cheapest price is penalised. Larger values result in less willingness to buy more expensive products.
- `LOYALTY_MULTIPLIER` - The score of the preferred manufacturer is multiplied by this.
- `RIPOFF_MULTIPLIER` - The proportion more than the cheapest price after which the product will never be purchased. E.g. 0.5 means we never buy products 50% more expensive than cheapest
- `BASE_MARGINAL` - Modifies the starting price of each product.
- `PRICE_INCREMENT` - Set the daily change in price by manufacturers.
- `MAX_PRICE_MULTIPLIER` - Prices of products cannot exceed this value.

The following parameters can be set with `COMPUTE_ON_DEVICE` or `COMPUTE_ON_HOST`. Note that there are several versions of some device functions, which are commented out in the `launch_[function_name]` functions.
- `PRICE_RESPONSE_COMPUTE`
- `MODIFY_PRICE_COMPUTE`
- `UPDATE_LOYALTIES_COMPUTE`
- `CONSUMER_CHOICE_COMPUTE` - Device version of this is bugged.

Data Output
==================
After simulation completes, data files can be found in the `src/logs` directory. Running the `plot` script found in the `src` directory will plot the data without running the simulation.

The data that is output contains points for all data, but the plotting scripts will only plot the first two columns of data in each file.

A loyalty live plot is also available, but for large number of days the output is not particularly useful, so it has been disabled. To re-enable it, uncomment lines 12-15 in the `plot` script.