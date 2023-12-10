import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import time
import itertools
import argparse

def parse_args() -> argparse.Namespace:
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu1", type=str, default="gpu1", help="Label for gpu 1"
    )
    parser.add_argument(
        "--gpu2", type=str, default="gpu2", help="Label for gpu 2"
    )
    parser.add_argument(
        "--trace1", type=int, nargs="+", help="Path to trace file for gpu 1", required=True
    )
    parser.add_argument(
        "--trace2", type=int, nargs="+", help="Path to trace file for gpu 2", required=True
    )

    return parser.parse_args()

# The below code loads the json data from the profiling runs and creates a dataframe with the data 

def load_json_data(filename): 
    with open(filename) as f:
        data = json.load(f)

    dataframe =  pd.DataFrame(columns=['batch_size', 'power_limit', 'time_per_epoch', 'average_power'])

    for k, val in data.items():
        for v in val:
            dataframe = dataframe.append({'batch_size' : int(k), 'power_limit': v['power_limit'], 'time_per_epoch': v['time'], 'average_power': v['energy'] / v['time']}, ignore_index=True)

    return dataframe

# The below code fits a curve to the data and plots the curve on the same graph as the data
# The function takes in a dataframe and a string that is the name of the gpu that the data is from
# The function returns two dataframes with the fitted parameters for the time per epoch and average power equations
# The function also plots the data and the fitted curve on the same graph

def fitcurveFunc(df, gpu):
    
    # One color per power limit
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    time_per_epoch_df = []
    

    power_limits = df['power_limit'].unique()

    # Loop through each power limit and create a curve for each:
    for i, p in enumerate(power_limits):

        x_values = df[df['power_limit'] == p]['batch_size'].values
        y_values = df[df['power_limit'] == p]['time_per_epoch'].values
        # Sort the values but keep the points together
        x_values, y_values = zip(*sorted(zip(x_values, y_values)))
        x_values = np.array(x_values)
        y_values = np.array(y_values)

        # Define the function to fit (sum of exponentials)
        def func(x, c,  d, e):
            return   c * x**2 + d * x + e

        # Make a guess for the parameters
        # guess = (1, -.05,  100, 20)
        # Fit the function to the data
        params, covariance = curve_fit(func, x_values, y_values, maxfev=100000)

        # Generate y values for the fitted curve
        y_fit = func(x_values, *params)

        time_per_epoch_df.append(params)

        # Plot the original points and the fitted curve on the same graph
        # Vary the color depending on the power limit
        plt.scatter(x_values, y_values, color = colors[i], label = "p_lim =" + str(p))
        plt.plot(x_values, y_fit, color=colors[i])
        plt.title('TimePerEpoch vs batch size power limit, ' + gpu)
        plt.xlabel('x')
        plt.ylabel('y')
        # Plot the legend in the uppr right corner
        plt.legend(loc='upper right')

        # Print the parameters of the fitted function
        print('Fitted Parameters:', params)
    plt.show()

    avg_power_df = []

    # Loop through each power limit and create a curve for each:
    for i, p in enumerate(power_limits):

        x_values = df[df['power_limit'] == p]['batch_size'].values
        y_values = df[df['power_limit'] == p]['average_power'].values
        # Sort the values but keep the points together
        x_values, y_values = zip(*sorted(zip(x_values, y_values)))
        x_values = np.array(x_values)
        y_values = np.array(y_values)

        # Define the function to fit (sum of exponentials)
        def func(x, a, b, c,  d):
            return a * np.exp(b * x + c) + d

        # Make a guess for the parameters
        guess = (-1, -.05,  100, 20)
        # Fit the function to the data
        params, covariance = curve_fit(func, x_values, y_values, guess, maxfev=100000)

        # Generate y values for the fitted curve
        y_fit = func(x_values, *params)

        avg_power_df.append(params)

        # Plot the original points and the fitted curve on the same graph
        # Vary the color depending on the power limit
        plt.scatter(x_values, y_values, color = colors[i], label = "p_lim =" + str(p))
        plt.plot(x_values, y_fit, color=colors[i])
        plt.title('Average power vs batch size power limit, ' + gpu)
        plt.xlabel('x')
        plt.ylabel('y')
        # Plot the legend in the uppr right corner
        plt.legend(loc='upper right')

            # Print the parameters of the fitted function
        print('Fitted Parameters:', params)
    plt.show()

    return time_per_epoch_df, avg_power_df

# The below code finds the optimal allocaton of global batch size 4096 between the two GPUs
# The function takes in:
# - The fitted parameters for the time per epoch and average power equations for both GPUs
# - The power limits for both GPUs
# - The GPU strengths for both GPUs-- we assume that the lower ranked gpu (ie 1 as opposed to 2) is the stronger GPU
# - A boolean that is true if we want to find the optimal allocation for the naive version of the algorithm or false if we use the heuristic of allocatin weakest power limit to the strongest GPU
# The function returns a dictionary with the optimal allocation and the time it took to run the function

def findOptimalAllocation(power_limits_df1, power_limits_df2, gpuStrengths, naiveVersion, time_per_epoch_df1, time_per_epoch_df2, avg_power_df1, avg_power_df2):
    # Start time 
    start_time = time.time()
    # Generate batch sizes from the above equations:
    batchsizes = range(0, 4096, 1)
    
    batch_and_power_df1 = []
    batch_and_power_df2= []

    # Find the index of the in GPU strengths:
    maxGPU = gpuStrengths.index(min(gpuStrengths))

    def fillBatchAndPower(gpuType, isMaximum, naiveVersion):
        if gpuType == 0:
            if not naiveVersion and isMaximum:
                minPower = min(power_limits_df1)
                for b in batchsizes:
                    batch_and_power_df1.append((b, list(power_limits_df1).index(minPower)))
            else:
                for b in batchsizes:
                    for i in range(len(power_limits_df1)):
                        batch_and_power_df1.append((b, i))
        elif gpuType == 1:
            if not naiveVersion and isMaximum:
                minPower = min(power_limits_df2)
                for b in batchsizes:
                    batch_and_power_df2.append((b, list(power_limits_df2).index(minPower)))
            else:
                for b in batchsizes:
                    for i in range(len(power_limits_df2)):
                        batch_and_power_df2.append((b, i))
    
    fillBatchAndPower(0, 0 == maxGPU, naiveVersion)
    fillBatchAndPower(1, 1 == maxGPU, naiveVersion)

    print(batch_and_power_df1)
    results5 = {}
    global_batch_size = 4096
    for b1, p1 in batch_and_power_df1:
        for b2, p2 in batch_and_power_df2:
            b = [b1, b2]
            if sum(b) == global_batch_size:
                result = 0
                for i, batch in enumerate(b):
                    if i % 4 == 0:
                        time_per_epoch = time_per_epoch_df1[p1][0] * batch**2 + time_per_epoch_df1[p1][1] * batch + time_per_epoch_df1[p1][2]
                        if time_per_epoch < 0:
                            time_per_epoch = 0 
                        average_power = avg_power_df1[p1][0] * np.exp(avg_power_df1[p1][1] * batch + avg_power_df1[p1][2]) + avg_power_df1[p1][3]
                        if average_power < 0:
                            average_power = 0
                        
                        power_limit = power_limits_df1[p1]
                    elif i % 4 == 1:
                        time_per_epoch = time_per_epoch_df2[p2][0] * batch**2 + time_per_epoch_df2[p2][1] * batch + time_per_epoch_df2[p2][2]
                        if time_per_epoch < 0:
                            time_per_epoch = 0
                        average_power = avg_power_df2[p2][0] * np.exp(avg_power_df2[p2][1] * batch + avg_power_df2[p2][2]) + avg_power_df2[p2][3]
                        if average_power < 0:
                            average_power = 0
                        power_limit = power_limits_df2[p2]
                    # Calculate the results:
                    result += (.5 * average_power + .5 * power_limit) * time_per_epoch
                powerlimit1 = power_limits_df1[p1]
                powerlimit2 = power_limits_df2[p2]
                results5[tuple([(b1, powerlimit1), (b2, powerlimit2)])] = result
    # Stop the time 
    end_time = time.time()
    return results5, end_time - start_time

# The below code runs the simulation for all combinations of the GPUs
# The function takes in:
# - The names of the GPUs
# - The power limits for both GPUs
# - The fitted parameters for the time per epoch and average power equations for both GPUs
# - The GPU strengths for both GPUs-- we assume that the lower ranked gpu (ie 1 as opposed to 2) is the stronger GPU
# The function returns a dataframe with the results of the simulation

def runSimulation(gpuNames, gpuPowerLimits, avg_power_dfs, time_per_epoch_dfs, gpuStrengths):
    finalResults = pd.DataFrame(columns=['gpu1', 'gpu2', 'time', 'naive', 'cost', 'topAllocation', 'maxCost'])
    # Get all combinations of the GPUs
    for i1, i2 in itertools.combinations(range(len(gpuNames)), 2):
        gpuStrength = [gpuStrengths[i1], gpuStrengths[i2]]
        # Get the gpu names
        gpu1 = gpuNames[i1]
        gpu2 = gpuNames[i2]
        # Get the time per epoch and average power for each of the GPUs
        time_per_epoch_df1 = time_per_epoch_dfs[i1]
        time_per_epoch_df2 = time_per_epoch_dfs[i2]

        avg_power_df1 = avg_power_dfs[i1]
        avg_power_df2 = avg_power_dfs[i2]

        power_limits_df1 = gpuPowerLimits[i1]
        power_limits_df2 = gpuPowerLimits[i2]

        # Run the simulation for the naive and non naive versions
        results5, time_taken = findOptimalAllocation(power_limits_df1, power_limits_df2, gpuStrength, True, time_per_epoch_df1, time_per_epoch_df2, avg_power_df1, avg_power_df2)
        finalResults = finalResults.append({'gpu1': gpu1, 'gpu2': gpu2, 'time': time_taken, 'naive': True, 'cost': min(results5.values()), 'topAllocation': sorted(results5.items(), key=lambda item: item[1])[0], 'maxCost': max(results5.values())}, ignore_index=True)
        results5, time_taken = findOptimalAllocation(power_limits_df1, power_limits_df2, gpuStrength, False, time_per_epoch_df1, time_per_epoch_df2, avg_power_df1, avg_power_df2)
        finalResults = finalResults.append({'gpu1': gpu1, 'gpu2': gpu2, 'time': time_taken, 'naive': False, 'cost': min(results5.values()), 'topAllocation': sorted(results5.items(), key=lambda item: item[1])[0], 'maxCost': max(results5.values())}, ignore_index=True)

        print("Iteration completed")
    return finalResults   

# The below code calculates the baseline results for the two GPUs where we assume that the batches are split evenly between the GPUs
# The function takes in:
# - The names of the GPUs
# - The power limits for both GPUs
# - The fitted parameters for the time per epoch and average power equations for both GPUs
# - The GPU strengths for both GPUs-- we assume that the lower ranked gpu (ie 1 as opposed to 2) is the stronger GPU
# The function returns a dataframe with the results of the simulation
def calculateBaseline(gpuNames, gpuPowerLimits, avg_power_dfs, time_per_epoch_dfs, gpuStrengths):
    baselineResults = pd.DataFrame(columns=['gpu1', 'gpu2', 'time', 'naive', 'cost', 'top20List'])
    global_batch_size = 4096
    # Get the baseline results for each of the GPUs if batches were split evenly between the GPUs
    for i1, i2 in itertools.combinations(range(len(gpuNames)), 2):
        # Get the gpu names
        gpu1 = gpuNames[i1]
        gpu2 = gpuNames[i2]
        # Get the time per epoch and average power for each of the GPUs
        time_per_epoch_df1 = time_per_epoch_dfs[i1]
        time_per_epoch_df2 = time_per_epoch_dfs[i2]

        avg_power_df1 = avg_power_dfs[i1]
        avg_power_df2 = avg_power_dfs[i2]

        power_limits_df1 = gpuPowerLimits[i1]
        power_limits_df2 = gpuPowerLimits[i2]

        tempResults = {}
        
        for p1 in range(len(power_limits_df1)):
            for p2 in range(len(power_limits_df2)):
                b = [2048, 2048]
                if sum(b) == global_batch_size:
                    result = 0
                    for i, batch in enumerate(b):
                        if i % 4 == 0:
                            time_per_epoch = time_per_epoch_df1[p1][0] * batch**2 + time_per_epoch_df1[p1][1] * batch + time_per_epoch_df1[p1][2]
                            average_power = avg_power_df1[p1][0] * np.exp(avg_power_df1[p1][1] * batch + avg_power_df1[p1][2]) + avg_power_df1[p1][3]
                            power_limit = power_limits_df1[p1]
                        elif i % 4 == 1:
                            time_per_epoch = time_per_epoch_df2[p2][0] * batch**2 + time_per_epoch_df2[p2][1] * batch + time_per_epoch_df2[p2][2]
                            average_power = avg_power_df2[p2][0] * np.exp(avg_power_df2[p2][1] * batch + avg_power_df2[p2][2]) + avg_power_df2[p2][3]
                            power_limit = power_limits_df2[p2]
                        
                    # Calculate the results:
                    result += (.5 * average_power + .5 * power_limit) * time_per_epoch
                powerlimit1 = power_limits_df1[p1]
                powerlimit2 = power_limits_df2[p2]
                tempResults[tuple([(2048, powerlimit1), (2048, powerlimit2)])] = result
    
    baselineResults = baselineResults.append({'gpu1': gpu1, 'gpu2': gpu2, 'time': 0, 'naive': False, 'cost': min(tempResults.values()), 'topAllocation': sorted(tempResults.items(), key=lambda item: item[1])[0]}, ignore_index=True)
    return baselineResults

def main(args: argparse.Namespace) -> None:
    # Create entries in a df with the key (batch_size) as a column with the other values in the columns
    gpu1_data = load_json_data(args.trace1)
    gpu2_data = load_json_data(args.trace2)

    # Fit the curves for the two dataframes
    time_per_epoch_gpu1, avg_power_gpu1 = fitcurveFunc(gpu1_data, args.gpu1)
    time_per_epoch_gpu2, avg_power_gpu2 = fitcurveFunc(gpu2_data, args.gpu2)

    # Run the simulation for the two given GPUs
    gpuNames = [args.gpu1, args.gpu1]
    gpuPowerLimits = [gpu1_data["power_limit"].unique(), gpu2_data["power_limit"].unique()]
    avg_power_dfs = [avg_power_gpu1, avg_power_gpu2]
    time_per_epoch_dfs = [time_per_epoch_gpu1, time_per_epoch_gpu2]
    gpuStrengths = [1,2]
    finalResults = runSimulation(gpuNames, gpuPowerLimits, avg_power_dfs, time_per_epoch_dfs, gpuStrengths)

    # Save the results to a csv file
    finalResults.to_csv(f'{args.gpu1}_{args.gpu2}_results.csv')

    # Run the baseline for the two given GPUs
    baselineResults = calculateBaseline(gpuNames, gpuPowerLimits, avg_power_dfs, time_per_epoch_dfs, gpuStrengths)

    # Save the results to a csv file
    baselineResults.to_csv(f'{args.gpu1}_{args.gpu2}_baseline.csv')

if __name__ == "__main__":
    main(parse_args())