import pandas as pd

# drop first column, repeat of index
all_matches = pd.read_excel("all_matches.xlsx")
all_matches = all_matches.drop(axis = 1, columns=['Unnamed: 0'])

# drop time from the DateTime column
all_matches['Date'] = pd.to_datetime(all_matches['Date']).dt.date

# drop incomplete games
all_matches = all_matches.dropna(subset=['W1', 'Wsets', 'W2'])
unfinished_games = all_matches.loc[all_matches['Wsets'].isin([1, 0])]
all_matches = pd.concat([all_matches, unfinished_games, unfinished_games]).drop_duplicates(keep=False)




