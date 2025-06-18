import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# Path to the folder containing CSV files
ac_results_folder = '/Users/leonardlogaric/Desktop/quera_proj/ac_results'
print("Looking for CSV files in:", ac_results_folder)
total_T, T_steps = 2.0, 200
times = np.linspace(0.0, total_T, T_steps)

# Find all CSV files in the folder
csv_files = glob.glob(os.path.join(ac_results_folder, '*.csv'))

if not csv_files:
    print("No CSV files found in ac_results folder.")
    exit(1)

# Read all CSV files into a list of numpy arrays
results = [np.genfromtxt(csv_file, delimiter=',') for csv_file in csv_files]
results = np.array(results)

# Computing average over trials
average_results = np.mean(results, axis=0)

# Plotting the average results against time
plt.plot(times, average_results.T)
plt.xlabel('Time')
plt.ylabel('Autocorrelator')
plt.title('Average Autocorrelator Over Trials')
plt.ylim(0.0, 1.0)  # Adjust y-axis limits as needed
plt.show()
