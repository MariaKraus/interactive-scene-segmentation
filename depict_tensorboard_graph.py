import pandas as pd
import matplotlib.pyplot as plt

name = 'pretrained_no_augmentation_Sep06_19-08-04_kraus'
# Read the CSV data
data = pd.read_csv(f'result_runs/{name}.csv')

# Apply a smoothing factor using a rolling window
smoothing_factor = 50  # Adjust this value as needed
data['Smoothed Value'] = data['Value'].rolling(window=smoothing_factor).mean()

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(data['Step'], data['Value'], label='Original Value')
plt.plot(data['Step'], data['Smoothed Value'], label='Smoothed Value', linewidth=2)
plt.xlabel('Step')
plt.ylabel('Avg Absolute Error/ train')
plt.title('Tensorboard Data with Smoothing')
plt.legend()
plt.grid(True)


# Adjust y-axis limits
lower_limit = 0  # Adjust this value as needed
upper_limit = 5  # Adjust this value as needed
plt.ylim(lower_limit, upper_limit)

plt.savefig(f'result_runs/{name}.png')
plt.show()
