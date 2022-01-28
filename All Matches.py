import pandas as pd

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

betting_odds1 = ['B365W', 'B365L', 'EXW', 'EXL', 'LBW', 'LBL', 'PSW', 'PSL', 'SJW', 'SJL', 'MaxW', 'MaxL', 'AvgW', 'AvgL']
betting_odds2 = ['B365W', 'B365L', 'EXW', 'EXL', 'LBW', 'LBL', 'PSW', 'PSL', 'MaxW', 'MaxL', 'AvgW', 'AvgL']
betting_odds3 = ['B365W', 'B365L', 'PSW', 'PSL', 'MaxW', 'MaxL', 'AvgW', 'AvgL']

matches_2010 = matches_2010.drop(columns = betting_odds1)
matches_2011 = matches_2011.drop(columns = betting_odds1)
matches_2012 = matches_2012.drop(columns = betting_odds1)
matches_2013 = matches_2013.drop(columns = betting_odds1)
matches_2014 = matches_2014.drop(columns = betting_odds1)
matches_2015 = matches_2015.drop(columns = betting_odds2)
matches_2016 = matches_2016.drop(columns = betting_odds2)
matches_2017 = matches_2017.drop(columns = betting_odds2)
matches_2018 = matches_2018.drop(columns = betting_odds2)
matches_2019 = matches_2019.drop(columns = betting_odds3)
matches_2020 = matches_2020.drop(columns = betting_odds3)
matches_2021 = matches_2021.drop(columns = betting_odds3)

all_matches = pd.concat([matches_2010, matches_2011, matches_2012, matches_2013, matches_2014, matches_2015, matches_2016, matches_2017, matches_2018, matches_2019, matches_2020, matches_2021], ignore_index = True)

