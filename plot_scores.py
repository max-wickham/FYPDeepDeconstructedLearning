
import pandas as pd
import matplotlib.pyplot as plt


location = 'ppo_9_stats'
for location in  ['ppo_8_stats','ppo_9_stats']:
    with open(f'{location}/scores.csv', 'r', encoding='utf-8') as file:
        lines = file.read().split('\n')[:-1]

    scores = [int(line) for line in lines]

    window_size = 100

    # Convert array of integers to pandas series
    numbers_series = pd.Series(scores)

    # Get the window of series
    # of observations of specified window size
    windows = numbers_series.rolling(window_size)

    # Create a series of moving
    # averages of each window
    moving_averages = windows.mean()

    moving_averages.plot(kind = 'line')
plt.show()
# # Convert pandas series back to list
# moving_averages_list = moving_averages.tolist()

# # Remove null entries from the list
# final_list = moving_averages_list[window_size - 1:]

# print(final_list)
