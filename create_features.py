import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random

start_time = time.time()


def data_change(data, surface, sets):

    data = data[(data['Comment'] == 'Completed')].reset_index(drop=True)

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


def choose_player(data):

    def variable_definition():

        winner = data['Winner']
        loser = data['Loser']
        winner_rank = data['WRank']
        loser_rank = data['LRank']
        winner_set_1 = data['W1']
        winner_set_2 = data['W2']
        winner_set_3 = data['W3']
        winner_set_4 = data['W4']
        winner_set_5 = data['W5']
        loser_set_1 = data['L1']
        loser_set_2 = data['L2']
        loser_set_3 = data['L3']
        loser_set_4 = data['L4']
        loser_set_5 = data['L5']

        player_0 = []
        player_1 = []
        outcome = []
        player_0_rank = []
        player_1_rank = []
        player_0_set_1 = []
        player_0_set_2 = []
        player_0_set_3 = []
        player_0_set_4 = []
        player_0_set_5 = []
        player_1_set_1 = []
        player_1_set_2 = []
        player_1_set_3 = []
        player_1_set_4 = []
        player_1_set_5 = []

        winner_sets = [winner_set_1, winner_set_2, winner_set_3, winner_set_4, winner_set_5]
        loser_sets = [loser_set_1, loser_set_2, loser_set_3, loser_set_4, loser_set_5]
        player_0_sets = [player_0_set_1, player_0_set_2, player_0_set_3, player_0_set_4, player_0_set_5]
        player_1_sets = [player_1_set_1, player_1_set_2, player_1_set_3, player_1_set_4, player_1_set_5]

        return winner, loser, winner_rank, loser_rank, player_0, player_1, player_0_rank, player_1_rank, outcome, winner_sets, loser_sets, player_0_sets, player_1_sets

    winner, loser, winner_rank, loser_rank, player_0, player_1, player_0_rank, player_1_rank, outcome, winner_sets, loser_sets, player_0_sets, player_1_sets = variable_definition()

    for i in range(data.shape[0]):

        n = random.randint(0, 1)

        if n == 0:
            player_0.append(winner[i])
            player_1.append(loser[i])
            outcome.append(1)
            player_0_rank.append(winner_rank[i])
            player_1_rank.append(loser_rank[i])

            for j in range(5):
                player_0_sets[j] = winner_sets[j]
                player_1_sets[j] = loser_sets[j]

        else:
            player_0.append(loser[i])
            player_1.append(winner[i])
            outcome.append(0)
            player_1_rank.append(winner_rank[i])
            player_0_rank.append(loser_rank[i])

            for j in range(5):
                player_0_sets[j] = loser_sets[j]
                player_1_sets[j] = winner_sets[j]

    data['player_0'] = player_0
    data['player_1'] = player_1
    data['outcome'] = outcome
    data['player_0_rank'] = player_0_rank
    data['player_1_rank'] = player_1_rank

    for i in range(5):
        data['player_0_set' + '_' + str(i+1)] = player_0_sets[i]
        data['player_1_set' + '_' + str(i+1)] = player_1_sets[i]

    data = data.drop(labels = ['WPts', 'LPts', 'Wsets', 'Lsets', 'Court', 'Series', 'Winner', 'Loser', 'WRank', 'LRank', 'W1', 'W2', 'W3', 'W4', 'W5', 'L1', 'L2', 'L3', 'L4', 'L5'], axis = 1)

    return data


def diff_rank(data):
    rank_0 = data['player_0_rank']
    rank_1 = data['player_1_rank']
    list_diff_rank = []

    for i in range(data.shape[0]):
        list_diff_rank.append(rank_0[i] - rank_1[i])

    data['diff_rank'] = list_diff_rank

    return data


def win_percentage(data, player, surface, sets, opponent):
    new_matches = data_change(data, surface, sets)

    if opponent != 'All':
        wins = ((new_matches['player_0'] == player) & (new_matches['player_1'] == opponent) & (new_matches['outcome'] == 1) | (new_matches['player_1'] == player) & (new_matches['player_0'] == opponent) & (new_matches['outcome'] == 0)).sum()
        losses = ((new_matches['player_0'] == player) & (new_matches['player_1'] == opponent) & (new_matches['outcome'] == 0) | (new_matches['player_1'] == player) & (new_matches['player_0'] == opponent) & (new_matches['outcome'] == 1)).sum()

    else:
        wins = ((new_matches['player_0'] == player) & (new_matches['outcome'] == 1) | (new_matches['player_1'] == player) & (new_matches['outcome'] == 0)).sum()
        losses = ((new_matches['player_1'] == player) & (new_matches['outcome'] == 1) | (new_matches['player_0'] == player) & (new_matches['outcome'] == 0)).sum()

    player_win_percentage = wins / (wins + losses)

    if wins + losses == 0:
        player_win_percentage = np.nan

    return player_win_percentage


def game_win_percentage(data, player, surface, sets, opponent):

    new_matches = data_change(data, surface, sets)

    if opponent != 'All':
        player_pt1 = new_matches[((new_matches['player_0'] == player) & (new_matches['player_1'] == opponent))].reset_index(drop=True)
        player_pt2 = new_matches[((new_matches['player_1'] == player) & (new_matches['player_0'] == opponent))].reset_index(drop=True)

    else:
        player_pt1 = new_matches[(new_matches['player_0'] == player)].reset_index(drop=True)
        player_pt2 = new_matches[(new_matches['player_1'] == player)].reset_index(drop=True)

    games_won = 0
    games_lost = 0

    player_0_sets = {'player_0_set_1', 'player_0_set_2', 'player_0_set_3', 'player_0_set_4', 'player_0_set_5'}
    player_1_sets = {'player_1_set_1', 'player_1_set_2', 'player_1_set_3', 'player_1_set_4', 'player_1_set_5'}

    for i in player_0_sets:
        games_won += player_pt1[i].sum()
        games_lost += player_pt2[i].sum()
    for j in player_1_sets:
        games_lost += player_pt1[j].sum()
        games_won += player_pt2[j].sum()

    player_game_win_percentage = games_won / (games_won + games_lost)

    if games_won + games_lost == 0:
        player_game_win_percentage = np.nan

    return player_game_win_percentage


def diff_generator(data, surface, sets, opponent, type):
    new_matches = data_change(data, surface, sets)

    player_0 = new_matches['player_0']
    player_1 = new_matches['player_1']

    list_diff = []

    for i in range(0, new_matches.shape[0]):
        use_matches = new_matches.iloc[0:i, :]

        if type == 'match':
            if opponent != 'All':
                player_0_win_percentage = win_percentage(use_matches, player_0[i], surface, sets, player_1[i])
                player_1_win_percentage = win_percentage(use_matches, player_1[i], surface, sets, player_0[i])

            else:
                player_0_win_percentage = win_percentage(use_matches, player_0[i], surface, sets, opponent)
                player_1_win_percentage = win_percentage(use_matches, player_1[i], surface, sets, opponent)

            list_diff.append(player_0_win_percentage - player_1_win_percentage)

        if type == 'game':
            if opponent != 'All':
                player_0_game_win_percentage = game_win_percentage(use_matches, player_0[i], surface, sets, player_1[i])
                player_1_game_win_percentage = game_win_percentage(use_matches, player_1[i], surface, sets, player_0[i])

            else:
                player_0_game_win_percentage = game_win_percentage(use_matches, player_0[i], surface, sets, opponent)
                player_1_game_win_percentage = game_win_percentage(use_matches, player_1[i], surface, sets, opponent)

            list_diff.append(player_0_game_win_percentage - player_1_game_win_percentage)

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

    new_matches_sets_combined = pd.merge(new_matches_opponent, new_matches_sets_combined)
    new_matches_sets_combined = pd.merge(new_matches_win, new_matches_sets_combined)
    new_matches_sets_combined = pd.merge(new_matches_sets_combined, new_matches_surfaces_combined)
    new_matches_sets_combined = pd.merge(new_matches_sets_combined, new_matches_surfaces_sets_combined)
    new_matches_game_sets_combined = pd.merge(new_matches_game_opponent, new_matches_game_sets_combined)
    new_matches_game_sets_combined = pd.merge(new_matches_game_win, new_matches_game_sets_combined)
    new_matches_game_sets_combined = pd.merge(new_matches_game_sets_combined, new_matches_game_surfaces_combined)
    new_matches_game_sets_combined = pd.merge(new_matches_game_sets_combined, new_matches_game_surfaces_sets_combined)
    full_features = pd.merge(new_matches_sets_combined, new_matches_game_sets_combined)

    return full_features


matches = pd.read_csv('allmatches.csv')
matches = choose_player(matches)

rem_features = ['B365W', 'B365L', 'EXW', 'EXL', 'LBW', 'LBL', 'PSW',
                'PSL', 'SJW', 'SJL', 'MaxW', 'MaxL', 'AvgW', 'AvgL']

for feature in rem_features:
    matches = matches.drop(feature, axis=1)

diff_features = create_diff(matches)
diff_features.to_csv('./diff_features.csv')


# print(feature_combiner_game(matches, 'All', 'All', 'All'))


print("--- %s seconds ---" % (time.time() - start_time))
