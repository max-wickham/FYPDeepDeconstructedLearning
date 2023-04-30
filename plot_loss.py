import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ACTOR = True
LOCATION = 'ppo_9_stats'

with open(f'{LOCATION}/loss_stats.csv', 'r', encoding='utf-8') as file:
    lines = file.read().split('\n')[2:-1]

scores = [
    [float(line.split(',')[i]) for line in lines]
    for i in range(len(lines[0].split(',')))
    ]
scores = [
    [x if not math.isnan(x) else 0 for x in score ]
    for score in scores
]
scores = [
    np.array(score) / (np.max(score) + 1) for score in scores
]



# actor_scores = [
#     [float(line.split(',')[i]) for line in lines]
#     for i in range(len(line.split(','))-1)
#     ]
# critic_scores = [float(line.split(',')[-1]) for line in lines]

# critic_scores = np.array(critic_scores) / np.max(critic_scores)
window_size = 1
print(scores)
for score in scores[:-1]:

    # Convert array of integers to pandas series
    numbers_series = pd.Series(score)

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
