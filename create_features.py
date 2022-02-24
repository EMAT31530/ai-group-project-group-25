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

    else:
        if sets == '3':
            data = data[data['Best of'] == 3].reset_index(drop=True)

        if sets == '5':
            data = data[data['Best of'] == 5].reset_index(drop=True)

    return data


def diff_rank(data):
    rank_loser = data['LRank']
    rank_winner = data['WRank']
    list_diff_rank = []

    for i in range(data.shape[0]):
        list_diff_rank.append(abs(rank_winner[i] - rank_loser[i]))

    data['diff_rank'] = list_diff_rank

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


def game_win_percentage(data, player, surface, sets, opponent):
    data = data[(data['Winner'] == player) | (data['Loser'] == player)].reset_index(drop=True)
    new_matches = data_change(data, surface, sets)

    games_won = 0
    games_lost = 0

    winner_set_number = {'W1', 'W2', 'W3', 'W4', 'W5'}
    loser_set_number = {'L1', 'L2', 'L3', 'L4', 'L5'}

    if opponent != 'All':
        winner_data = new_matches[(new_matches['Winner'] == player) & (new_matches['Loser'] == opponent)]
        loser_data = new_matches[(new_matches['Loser'] == player) & (new_matches['Winner'] == opponent)]

    else:
        winner_data = new_matches[new_matches['Winner'] == player]
        loser_data = new_matches[new_matches['Loser'] == player]

    for i in winner_set_number:
        games_won += winner_data[i].sum()
        games_lost += loser_data[i].sum()
    for j in loser_set_number:
        games_lost += winner_data[j].sum()
        games_won += loser_data[j].sum()

    player_game_win_percentage = games_won / (games_won + games_lost)

    if games_won + games_lost == 0:
        player_game_win_percentage = np.nan

    return player_game_win_percentage


def diff_generator(data, surface, sets, opponent, type):
    new_matches = data_change(data, surface, sets)
    winner = new_matches['Winner']
    loser = new_matches['Loser']

    list_diff = []

    for i in range(0, new_matches.shape[0]):
        use_matches = new_matches.iloc[0:i, :]

        if type == 'match':
            if opponent != 'All':
                player_1_win_percentage = win_percentage(use_matches, winner[i], surface, sets, loser[i])
                player_2_win_percentage = win_percentage(use_matches, loser[i], surface, sets, winner[i])

            else:
                player_1_win_percentage = win_percentage(use_matches, winner[i], surface, sets, opponent)
                player_2_win_percentage = win_percentage(use_matches, loser[i], surface, sets, opponent)

            list_diff.append(player_1_win_percentage - player_2_win_percentage)

        if type == 'game':
            if opponent != 'All':
                player_1_game_win_percentage = game_win_percentage(use_matches, winner[i], surface, sets, loser[i])
                player_2_game_win_percentage = game_win_percentage(use_matches, loser[i], surface, sets, winner[i])

            else:
                player_1_game_win_percentage = game_win_percentage(use_matches, winner[i], surface, sets, opponent)
                player_2_game_win_percentage = game_win_percentage(use_matches, loser[i], surface, sets, opponent)

            list_diff.append(player_1_game_win_percentage - player_2_game_win_percentage)

    return new_matches, list_diff


def feature_combiner_match(data, surface, sets, opponent, type):
    surface_labels = {'Clay', 'Hard', 'Grass'}
    set_labels = {'3', '5'}

    if surface == sets == opponent == 'All':

        new_matches, list_diff = diff_generator(data, 'All', 'All', 'All', type)
        if type == 'game':
            new_matches['diff_game_win_percentage'] = list_diff
        if type == 'match':
            new_matches['diff_match_win_percentage'] = list_diff

        return new_matches

    if (opponent != 'All') & (sets == surface == 'All'):
        new_matches_opponent, list_diff_opponent = diff_generator(data, 'All', 'All', 'Yes', type)
        if type == 'game':
            new_matches_opponent['diff_game_win_percentage_hh'] = list_diff_opponent
        if type == 'match':
            new_matches_opponent['diff_match_win_percentage_hh'] = list_diff_opponent

        return new_matches_opponent

    if (surface != 'All') & (sets == opponent == 'All'):

        for i in surface_labels:
            globals()['new_matches_%s' % i], globals()['list_diff_%s' % i] = diff_generator(data, i, 'All', 'All', type)

            if type == 'match':
                globals()['new_matches_%s' % i]['diff_match_win_percentage_surface'] = globals()['list_diff_%s' % i]

            if type == 'game':
                globals()['new_matches_%s' % i]['diff_game_win_percentage_surface'] = globals()['list_diff_%s' % i]

        new_matches_surfaces_combined = pd.concat([new_matches_Grass, new_matches_Clay, new_matches_Hard])

        return new_matches_surfaces_combined

    if (sets != 'All') & (surface == opponent == 'All'):

        for i in set_labels:
            globals()['new_matches_%s' % i], globals()['list_diff_%s' % i] = diff_generator(data, 'All', i, 'All', type)

            if type == 'match':
                globals()['new_matches_%s' % i]['diff_match_win_percentage_sets'] = globals()['list_diff_%s' % i]

            if type == 'game':
                globals()['new_matches_%s' % i]['diff_game_win_percentage_sets'] = globals()['list_diff_%s' % i]

        new_matches_sets_combined = pd.concat([new_matches_3, new_matches_5])

        return new_matches_sets_combined

    if (sets != 'All') & (surface != 'All') & (opponent == 'All'):

        for i in set_labels:
            for j in surface_labels:
                globals()['new_matches_{0}_{1}'.format(i, j)], globals()[
                    'list_diff_{0}_{1}'.format(i, j)] = diff_generator(data, j, i, 'All', type)

                if type == 'match':
                    globals()['new_matches_{0}_{1}'.format(i, j)]['diff_match_win_percentage_surface_sets'] = globals()[
                        'list_diff_{0}_{1}'.format(i, j)]

                if type == 'game':
                    globals()['new_matches_{0}_{1}'.format(i, j)]['diff_game_win_percentage_surface_sets'] = globals()[
                        'list_diff_{0}_{1}'.format(i, j)]

        new_matches_surfaces_sets_combined = pd.concat(
            [new_matches_3_Grass, new_matches_3_Clay, new_matches_3_Hard, new_matches_5_Grass,
             new_matches_5_Clay, new_matches_5_Hard])

        return new_matches_surfaces_sets_combined


def create_diff(data):
    new_matches_rank = diff_rank(data)
    new_matches_win = feature_combiner_match(data, 'All', 'All', 'All', 'match')
    new_matches_opponent = feature_combiner_match(data, 'All', 'All', 'Yes', 'match')
    new_matches_surfaces_combined = feature_combiner_match(data, 'Yes', 'All', 'All', 'match')
    new_matches_sets_combined = feature_combiner_match(data, 'All', 'Yes', 'All', 'match')
    new_matches_surfaces_sets_combined = feature_combiner_match(data, 'Yes', 'Yes', 'All', 'match')

    new_matches_game_win = feature_combiner_match(data, 'All', 'All', 'All', 'game')
    new_matches_game_opponent = feature_combiner_match(data, 'All', 'All', 'Yes', 'game')
    new_matches_game_surfaces_combined = feature_combiner_match(data, 'Yes', 'All', 'All', 'game')
    new_matches_game_sets_combined = feature_combiner_match(data, 'All', 'Yes', 'All', 'game')
    new_matches_game_surfaces_sets_combined = feature_combiner_match(data, 'Yes', 'Yes', 'All', 'game')

    new_matches_sets_combined = pd.merge(new_matches_sets_combined, new_matches_surfaces_combined)
    new_matches_sets_combined = pd.merge(new_matches_sets_combined, new_matches_surfaces_sets_combined)
    new_matches_game_sets_combined = pd.merge(new_matches_game_sets_combined, new_matches_game_surfaces_combined)
    new_matches_game_sets_combined = pd.merge(new_matches_game_sets_combined, new_matches_game_surfaces_sets_combined)
    full_features = pd.merge(new_matches_sets_combined, new_matches_game_sets_combined)

    return full_features


matches = pd.read_csv('allmatches.csv')

rem_features = ['B365W', 'B365L', 'EXW', 'EXL', 'LBW', 'LBL', 'PSW',
                'PSL', 'SJW', 'SJL', 'MaxW', 'MaxL', 'AvgW', 'AvgL']

for feature in rem_features:
    matches = matches.drop(feature, axis=1)

new_matches = create_diff(matches)
# diff_features.to_csv('./diff_features.csv')

# print(feature_combiner_game(matches, 'All', 'All', 'All'))

print("--- %s seconds ---" % (time.time() - start_time))
