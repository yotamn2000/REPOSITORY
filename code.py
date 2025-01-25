import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calc_mean_erp(trial_points, ecog_data):
   """
   Calculates and plots the mean event-related potential (ERP) for each finger movement.
   
   Parameters:
   trial_points (str): CSV file with 3 columns:
       - Column 1: Starting point indices of finger movements
       - Column 2: Peak point indices of finger movements
       - Column 3: Finger numbers (1-5)
   ecog_data (str): CSV file with 1 column of brain signal time series
   
   Returns:
   np.ndarray: 5x1201 matrix where each row contains the mean ERP for a finger
   """
   try:
       # Load data
       trial_df = pd.read_csv(trial_points)
       ecog_df = pd.read_csv(ecog_data, usecols=[0])
       
       # Verify total trials
       total_trials = len(trial_df)
       if total_trials < 600:
           raise ValueError(f"Expected 600+ trials, but got {total_trials}")
       
       # Convert trial data to int
       trial_df = trial_df.astype({trial_df.columns[0]: int,  # Start indices
                                 trial_df.columns[1]: int,  # Peak indices
                                 trial_df.columns[2]: int}) # Finger numbers
       print("Successfully converted trial data to integer type")
       
       # Verify data synchronization
       max_start_idx = trial_df[trial_df.columns[0]].max() + 1001
       if len(ecog_df) < max_start_idx:
           raise ValueError("ECoG data indices don't match trial points")
       
       # Initialize output matrix (5 fingers x 1201 timepoints)
       fingers_erp_mean = np.zeros((5, 1201))
       time_points = np.arange(-200, 1001)
       
       # Process each finger
       for finger in range(1, 6):
           finger_trials = trial_df[trial_df.iloc[:, 2] == finger]
           start_indices = finger_trials.iloc[:, 0].values
           all_erp = np.vstack([ecog_df.iloc[idx-200:idx+1001, 0].values 
                               for idx in start_indices])
           fingers_erp_mean[finger-1, :] = np.mean(all_erp, axis=0)
       
       # Visualization
       plt.figure(figsize=(12, 6))
       for i in range(5):
           plt.plot(time_points, fingers_erp_mean[i], label=f'Finger {i+1}')
       
       plt.xlabel('Time (ms)')
       plt.ylabel('ERP Amplitude')
       plt.title('Mean ERP by Finger')
       plt.legend()
       plt.grid(True)
       plt.show()
       
       return fingers_erp_mean
       
   except Exception as e:
       print(f"Error occurred: {str(e)}")
       raise

if __name__ == "__main__":
   trial_points = "events_file_ordered.csv"
   ecog_data = "brain_data_channel_one.csv"
   
   erp_matrix = calc_mean_erp(trial_points, ecog_data)
   print(f"Dimensions of fingers_erp_mean: {erp_matrix.shape}")