import pandas as pd

# create a dataframe for each calendar year
matches_2010 = pd.read_excel('Full Tennis dataset.xls', sheet_name='2010')
matches_2011 = pd.read_excel('Full Tennis dataset.xls', sheet_name='2011')
matches_2012 = pd.read_excel('Full Tennis dataset.xls', sheet_name='2012')
matches_2013 = pd.read_excel('Full Tennis dataset.xls', sheet_name='2013')
matches_2014 = pd.read_excel('Full Tennis dataset.xls', sheet_name='2014')
matches_2015 = pd.read_excel('Full Tennis dataset.xls', sheet_name='2015')
matches_2016 = pd.read_excel('Full Tennis dataset.xls', sheet_name='2016')
matches_2017 = pd.read_excel('Full Tennis dataset.xls', sheet_name='2017')
matches_2018 = pd.read_excel('Full Tennis dataset.xls', sheet_name='2018')
matches_2019 = pd.read_excel('Full Tennis dataset.xls', sheet_name='2019')
matches_2020 = pd.read_excel('Full Tennis dataset.xls', sheet_name='2020')
matches_2021 = pd.read_excel('Full Tennis dataset.xls', sheet_name='2021')

sheet_names = [matches_2010, matches_2011, matches_2012, matches_2013, matches_2014, matches_2015, matches_2016, matches_2017, matches_2018, matches_2019, matches_2020, matches_2021]

# take off all the betting odds so that all the dataframes are in the same format
betting_odds1 = ['B365W', 'B365L', 'PSW', 'PSL', 'MaxW', 'MaxL', 'AvgW', 'AvgL']
betting_odds2 = ['B365W', 'B365L', 'PSW', 'PSL', 'MaxW', 'MaxL', 'AvgW', 'AvgL', 'EXW', 'EXL', 'LBW', 'LBL']
betting_odds3 = ['B365W', 'B365L', 'PSW', 'PSL', 'MaxW', 'MaxL', 'AvgW', 'AvgL', 'EXW', 'EXL', 'LBW', 'LBL', 'SJW', 'SJL']

for data in sheet_names[0:5]:
    data = data.drop(columns = betting_odds3)
for data in sheet_names[5:9]:
    data = data.drop(columns = betting_odds2)
for data in sheet_names[9:12]:
    data = data.drop(columns = betting_odds1)

# put all the dataframes into one large chronological dataframe, including all data from 2010-2021
all_matches = pd.concat(sheet_names, ignore_index = True)
all_matches.to_excel('all_matches.xlsx')