import pandas as pd
import numpy as np


def find(date):
    for i in dates.index:
        if dates[i] == date:
            return i

    return None


def games_won(player, date):
    ind_date = find(date)
    i = 0
    wins = 0
    while i < ind_date:
        if player == winner[i]:
            wins += 1
        i += 1
    return wins


def games_lost(player, date):
    ind_date = find(date)
    i = 0
    loses = 0
    while i < ind_date:
        if player == loser[i]:
            loses += 1
        i += 1
    return loses


def sets_won(player, date):
    ind_date = find(date)
    i = 0
    sets = 0
    while i < ind_date:
        if player == winner[i]:
            if np.isnan(set_winner[i]):
                sets += 0
            else:
                sets += set_winner[i]
        elif player == loser[i]:
            if np.isnan(set_loser[i]):
                sets += 0
            else:
                sets += set_loser[i]
        i += 1
    return sets


def sets_lost(player, date):
    ind_date = find(date)
    i = 0
    sets = 0
    while i < ind_date:
        if player == winner[i]:
            if np.isnan(set_loser[i]):
                sets += 0
            else:
                sets += set_loser[i]
        elif player == loser[i]:
            if np.isnan(set_winner[i]):
                sets += 0
            else:
                sets += set_winner[i]
        i += 1
    return sets


def max_rank(date):
    year = int(date[2:4])
    rank_year = year - 2
    ranks = []
    match_number = 0
    ind_date = find(date)
    while match_number < ind_date:
        year_match = int(dates_list[match_number][2:4])
        if year_match > rank_year:
            ranks = ranks + [rank_loser[match_number]]
            ranks = ranks + [rank_winner[match_number]]
        match_number += 1
    maximum_rank = max(ranks)
    return maximum_rank


def current_rank(player, date):
    match_number = 0
    rank = 'no previous'
    ind_date = find(date)
    while match_number < ind_date:
        no_match = ind_date - match_number
        if player == winner[no_match]:
            rank = rank_winner[no_match]
            return rank
        elif player == loser[no_match]:
            rank = rank_loser[no_match]
            return rank
        match_number += 1
    return rank


rem_features = ['B365W', 'B365L', 'EXW', 'EXL',
                'LBW', 'LBL', 'PSW', 'PSL', 'SJW', 'SJL', 'MaxW', 'MaxL', 'AvgW',
                'AvgL']

matches = pd.read_csv('allmatches.csv')

for feature in rem_features:
    matches = matches.drop(feature, 1)

dates = matches['Date']
dates_list = dates.to_list()
winner = matches['Winner']
loser = matches['Loser']
set_winner = matches['Wsets']
set_loser = matches['Lsets']
rank_loser = matches['LRank']
rank_winner = matches['WRank']
date = '2018-10-08'
player = 'Gasquet R.'




# print(matches['Date'])
# print(matches.columns)
# print(games_won(player, date))
# print(games_lost(player, date))
# print(sets_won(player, date))
# print(sets_lost(player, date))
print(current_rank(player, date))
