import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random

start_time = time.time()


# limit data depending on number of sets or type of surface required
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


# randomly associate winners and losers as players 0 and 1, rewrite features in terms of players 0 and 1
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
            outcome.append(0)
            player_0_rank.append(winner_rank[i])
            player_1_rank.append(loser_rank[i])

            for j in range(5):
                set_number_win = winner_sets[j]
                set_number_lose = loser_sets[j]
                player_0_sets[j].append(set_number_win[i])
                player_1_sets[j].append(set_number_lose[i])

        if n == 1:
            player_0.append(loser[i])
            player_1.append(winner[i])
            outcome.append(1)
            player_1_rank.append(winner_rank[i])
            player_0_rank.append(loser_rank[i])

            for j in range(5):
                set_number_win = winner_sets[j]
                set_number_lose = loser_sets[j]
                player_0_sets[j].append(set_number_lose[i])
                player_1_sets[j].append(set_number_win[i])

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


# create differences in ranks for all matches
def diff_rank(data):
    rank_0 = data['player_0_rank']
    rank_1 = data['player_1_rank']
    list_diff_rank = []

    for i in range(data.shape[0]):
        list_diff_rank.append(rank_0[i] - rank_1[i])

    data['diff_rank'] = list_diff_rank

    return data


# calculate individual match win percentage in terms of surface, sets and opponents
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
        player_win_percentage = 0

    return player_win_percentage


# calculate individual game win percentage in terms of surface, sets and opponents
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
        player_game_win_percentage = 0

    return player_game_win_percentage


# create list of differences in match/game win percentages for all relevant matches
def diff_generator(data, surface, sets, opponent, type, timeframe):
    new_matches = data_change(data, surface, sets)

    player_0 = new_matches['player_0']
    player_1 = new_matches['player_1']

    list_diff = []

    for i in range(0, new_matches.shape[0]):

        if timeframe == 'full':
            timestep = 0

        if timeframe == 'year':
            if i < 365:
                timestep = 0
            else:
                timestep = i - 365

        use_matches = new_matches.iloc[timestep:i, :]

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


# add lists as new column in data
def feature_combiner_match(data, surface, sets, opponent, type, timeframe):
    surface_labels = {'Clay', 'Hard', 'Grass'}
    set_labels = {'3', '5'}

    if surface == sets == opponent == 'All':

        new_matches, list_diff = diff_generator(data, 'All', 'All', 'All', type, timeframe)
        if type == 'game':
            if timeframe == 'year':
                new_matches['diff_game_win_percentage_year'] = list_diff
            else:
                new_matches['diff_game_win_percentage'] = list_diff
        if type == 'match':
            if timeframe == 'year':
                new_matches['diff_match_win_percentage_year'] = list_diff
            else:
                new_matches['diff_match_win_percentage'] = list_diff


        return new_matches

    if (opponent != 'All') & (sets == surface == 'All'):
        new_matches_opponent, list_diff_opponent = diff_generator(data, 'All', 'All', 'Yes', type, timeframe)
        if type == 'game':
            new_matches_opponent['diff_game_win_percentage_hh'] = list_diff_opponent
        if type == 'match':
            new_matches_opponent['diff_match_win_percentage_hh'] = list_diff_opponent

        return new_matches_opponent

    if (surface != 'All') & (sets == opponent == 'All'):

        for i in surface_labels:
            globals()['new_matches_{0}_{1}'.format(i, timeframe)], globals()['list_diff_{0}_{1}'.format(i, timeframe)] = diff_generator(data, i, 'All', 'All', type, timeframe)

            if type == 'match':
                if timeframe == 'year':
                    globals()['new_matches_{0}_{1}'.format(i, timeframe)]['diff_match_win_percentage_year_surface'] = globals()['list_diff_{0}_{1}'.format(i, timeframe)]
                else:
                    globals()['new_matches_{0}_{1}'.format(i, timeframe)]['diff_match_win_percentage_surface'] = globals()['list_diff_{0}_{1}'.format(i, timeframe)]

            if type == 'game':
                if timeframe == 'year':
                    globals()['new_matches_{0}_{1}'.format(i, timeframe)]['diff_game_win_percentage_year_surface'] = globals()['list_diff_{0}_{1}'.format(i, timeframe)]
                else:
                    globals()['new_matches_{0}_{1}'.format(i, timeframe)]['diff_game_win_percentage_surface'] = globals()['list_diff_{0}_{1}'.format(i, timeframe)]

        if timeframe == 'year':
            new_matches_surfaces_combined = pd.concat([new_matches_Grass_year, new_matches_Clay_year, new_matches_Hard_year])
        else:
            new_matches_surfaces_combined = pd.concat([new_matches_Grass_full, new_matches_Clay_full, new_matches_Hard_full])

        return new_matches_surfaces_combined

    if (sets != 'All') & (surface == opponent == 'All'):

        for i in set_labels:
            globals()['new_matches_%s' % i], globals()['list_diff_%s' % i] = diff_generator(data, 'All', i, 'All', type, timeframe)

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
                    'list_diff_{0}_{1}'.format(i, j)] = diff_generator(data, j, i, 'All', type, timeframe)

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


# combine all dataframes together
def create_diff(data):
    new_matches_rank = diff_rank(data)
    new_matches_win = feature_combiner_match(data, 'All', 'All', 'All', 'match', 'full')
    new_matches_win_year = feature_combiner_match(data, 'All', 'All', 'All', 'match', 'year')
    new_matches_opponent = feature_combiner_match(data, 'All', 'All', 'Yes', 'match', 'full')
    new_matches_surfaces_combined = feature_combiner_match(data, 'Yes', 'All', 'All', 'match', 'full')
    new_matches_year_surfaces_combined = feature_combiner_match(data, 'Yes', 'All', 'All', 'match', 'year')
    new_matches_sets_combined = feature_combiner_match(data, 'All', 'Yes', 'All', 'match', 'full')
    new_matches_surfaces_sets_combined = feature_combiner_match(data, 'Yes', 'Yes', 'All', 'match', 'full')

    new_matches_game_win = feature_combiner_match(data, 'All', 'All', 'All', 'game', 'full')
    new_matches_game_win_year = feature_combiner_match(data, 'All', 'All', 'All', 'game', 'year')
    new_matches_game_opponent = feature_combiner_match(data, 'All', 'All', 'Yes', 'game', 'full')
    new_matches_game_surfaces_combined = feature_combiner_match(data, 'Yes', 'All', 'All', 'game', 'full')
    new_matches_game_year_surfaces_combined = feature_combiner_match(data, 'Yes', 'All', 'All', 'game', 'year')
    new_matches_game_sets_combined = feature_combiner_match(data, 'All', 'Yes', 'All', 'game', 'full')
    new_matches_game_surfaces_sets_combined = feature_combiner_match(data, 'Yes', 'Yes', 'All', 'game', 'full')

    new_matches_sets_combined = pd.merge(new_matches_opponent, new_matches_sets_combined)
    new_matches_sets_combined = pd.merge(new_matches_win, new_matches_sets_combined)
    new_matches_sets_combined = pd.merge(new_matches_win_year, new_matches_sets_combined)
    new_matches_sets_combined = pd.merge(new_matches_sets_combined, new_matches_surfaces_combined)
    new_matches_sets_combined = pd.merge(new_matches_sets_combined, new_matches_year_surfaces_combined)
    new_matches_sets_combined = pd.merge(new_matches_sets_combined, new_matches_surfaces_sets_combined)
    new_matches_game_sets_combined = pd.merge(new_matches_game_opponent, new_matches_game_sets_combined)
    new_matches_game_sets_combined = pd.merge(new_matches_game_win, new_matches_game_sets_combined)
    new_matches_game_sets_combined = pd.merge(new_matches_game_win_year, new_matches_game_sets_combined)
    new_matches_game_sets_combined = pd.merge(new_matches_game_sets_combined, new_matches_game_surfaces_combined)
    new_matches_game_sets_combined = pd.merge(new_matches_game_sets_combined, new_matches_game_year_surfaces_combined)
    new_matches_game_sets_combined = pd.merge(new_matches_game_sets_combined, new_matches_game_surfaces_sets_combined)
    full_features = pd.merge(new_matches_sets_combined, new_matches_game_sets_combined)

    diff = new_matches_game_surfaces_combined['diff_game_win_percentage_surface'] - new_matches_game_year_surfaces_combined['diff_game_win_percentage_year_surface']

    return full_features, diff


matches = pd.read_csv('allmatches.csv')
matches = choose_player(matches)

rem_features = ['B365W', 'B365L', 'EXW', 'EXL', 'LBW', 'LBL', 'PSW',
                'PSL', 'SJW', 'SJL', 'MaxW', 'MaxL', 'AvgW', 'AvgL']

for feature in rem_features:
    matches = matches.drop(feature, axis=1)

diff_features, diff = create_diff(matches)
diff_features.to_csv('./diff_features.csv')


# print(feature_combiner_game(matches, 'All', 'All', 'All'))


print("--- %s seconds ---" % (time.time() - start_time))
