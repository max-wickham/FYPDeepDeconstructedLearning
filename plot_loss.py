
import pandas as pd
import matplotlib.pyplot as plt

ACTOR = True
LOCATION = 'ppo_8_stats'

with open(f'{LOCATION}/loss_stats.csv', 'r', encoding='utf-8') as file:
    lines = file.read().split('\n')[2:-1]

actor_scores = [float(line.split(',')[0]) for line in lines]
critic_scores = [float(line.split(',')[1]) for line in lines]
window_size = 1

for scores in (actor_scores, critic_scores):

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
