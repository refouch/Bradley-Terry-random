#################
## Get ligue 1 results
#######################

import pandas as pd
import numpy as np

def get_ligue_results(date):
    """
    date: format yy-1yy-2
    example: for results 2022-2023, input '22223'
    """

    url = f'https://raw.githubusercontent.com/datasets/football-datasets/refs/heads/main/datasets/ligue-1/season-{date}.csv'

    return pd.read_csv(url)


def split_season(ligue):

    ligue_aller = ligue[0:0].copy()
    ligue_retour = ligue[0:0].copy()
    recall = {}

    for _, row in ligue.iterrows():

        home, away = row['HomeTeam'], row['AwayTeam']
        
        if home not in recall:
            recall[home] = set()
        if away not in recall:
            recall[away] = set()

        if away not in recall[home]:
            ligue_aller.loc[len(ligue_aller)] = row
        else:
            ligue_retour.loc[len(ligue_retour)] = row
        
        recall[home].add(away)
        recall[away].add(home)

    return ligue_aller,ligue_retour


def total_wins(ligue):

    home_wins = ligue[ligue['FTR'] == 'H'].groupby('HomeTeam').size().reset_index(name='HomeWins').rename(columns={'HomeTeam': 'Team'})
    away_wins = ligue[ligue['FTR'] == 'A'].groupby('AwayTeam').size().reset_index(name='AwayWins').rename(columns={'AwayTeam': 'Team'})

    wins = pd.merge(home_wins,away_wins, on='Team')
    wins['TotalWins'] = wins['HomeWins'] + wins ['AwayWins']
    return wins


def victory_matrix(ligue):
    teams = sorted(set(ligue['HomeTeam']).union(set(ligue['AwayTeam'])))
    team_to_index = {team: i for i, team in enumerate(teams)}

    n = len(teams)
    X = np.zeros((n,n), dtype= int)

    for _, row in ligue.iterrows():
        home, away, winner = row['HomeTeam'], row['AwayTeam'], row['FTR']
        i, j = team_to_index[home],team_to_index[away]

        if winner == 'H':
            X[i,j] = 1

        else:
            X[j,i] = 1
            
    return X