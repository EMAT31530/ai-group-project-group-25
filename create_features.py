import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

start_time = time.time()


def data_change(data, surface, sets):
    if surface != 'All':
        data = data[data['Surface'] == surface].reset_index(drop=True)

    if sets == '3':
        data = data[data['Best of'] == 3].reset_index(drop=True)

    if sets == '5':
        data = data[data['Best of'] == 5].reset_index(drop=True)

    return data


def win_percentage(data, player, surface, sets, opponent):
    new_matches = data_change(data, surface, sets)

    if opponent != 'All':
        wins = ((new_matches['Winner'] == player) & (new_matches['Loser'] == opponent)).sum()
        losses = ((new_matches['Winner'] == opponent) & (new_matches['Loser'] == player)).sum()

    else:
        wins = (new_matches['Winner'] == player).sum()
        losses = (new_matches['Loser'] == player).sum()

    player_win_percentage = wins / (wins + losses)

    if wins + losses == 0:
        player_win_percentage = np.nan

    return player_win_percentage


def games_win_percentage(player, surface, sets, date_index):
    return no


def diff_rank(data):

    rank_loser = data['LRank']
    rank_winner = data['WRank']
    list_diff_rank = []

    for i in range(data.shape[0]):
        list_diff_rank.append(abs(rank_winner[i] - rank_loser[i]))

    data['diff_rank'] = list_diff_rank
    return data


def diff_generator(matches, surface, sets, opponent):
    new_matches = data_change(matches, surface, sets)
    winner = new_matches['Winner']
    loser = new_matches['Loser']

    list_diff = []

    for i in range(0, new_matches.shape[0]):
        use_matches = new_matches.iloc[0:i, :]
        player_1_win_percentage = win_percentage(use_matches, winner[i], surface, sets, opponent)
        player_2_win_percentage = win_percentage(use_matches, loser[i], surface, sets, opponent)
        list_diff.append(player_1_win_percentage - player_2_win_percentage)

    return new_matches, list_diff


matches = pd.read_csv('allmatches.csv')

rem_features = ['B365W', 'B365L', 'EXW', 'EXL', 'LBW', 'LBL', 'PSW',
                'PSL', 'SJW', 'SJL', 'MaxW', 'MaxL', 'AvgW', 'AvgL']

for feature in rem_features:
    matches = matches.drop(feature, axis=1)


print(create_diff(matches))

print("--- %s seconds ---" % (time.time() - start_time))
