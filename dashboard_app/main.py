from fastapi import FastAPI

app = FastAPI()

from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd
import numpy as np
from joblib import load

model_saved = load('model_nba.joblib')

def predict_games(home_team, away_team):
    gamefinder = leaguegamefinder.LeagueGameFinder(date_from_nullable='01/01/2021', league_id_nullable='00')
    games = gamefinder.get_data_frames()[0]
    games = games[['TEAM_NAME', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL', 'PLUS_MINUS']]

    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])

    msk_home=(games['TEAM_NAME']==home_team)
    games_30_home=games[msk_home].sort_values('GAME_DATE').tail(30)
    home_plus_minus=games_30_home['PLUS_MINUS'].mean()

    msk_away=(games['TEAM_NAME']==away_team)
    games_30_away=games[msk_away].sort_values('GAME_DATE').tail(30)
    away_plus_minus=games_30_away['PLUS_MINUS'].mean()

    games_diff = home_plus_minus - away_plus_minus

    predict_home_win=model_saved.predict(np.array([games_diff]))[0]

    predict_winning_probability=model_saved.predict_proba(np.array([games_diff]))[0][1]
    return {'result':int(predict_home_win),
            'win_probability':float(predict_winning_probability)}

@app.get("/predict_nba_winner/")
def predict_games_results(home_team,away_team):
    return predict_games(home_team,away_team)