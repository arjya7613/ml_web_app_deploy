# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 18:31:27 2024

@author: HP
"""


import numpy as np 
import pandas as pd
df=pd.read_csv('C:\\Users\\HP\\OneDrive\\Desktop\\hackathon\\player_performance_custom_data.csv')
import pickle
mod = pickle.load(open('C:\\Users\\HP\\OneDrive\\Desktop\\hackathon\\model.sav','rb'))


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

X = df[['mean_opponent_score', 'current_score_variability', 'kd_ratio', 'game_duration_remaining', 'win_rate']]
y = df['winning_probability']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mod=RandomForestRegressor(n_estimators=100,random_state=42)
mod.fit(X,y)
ypred = mod.predict(X_test)

def get_user_input():
    print("\nPlease enter the following player statistics:")
    mean_opponent_score = float(input("Mean Opponent Score: "))
    current_score_variability = float(input("Current Score Variability: "))
    kd_ratio = float(input("K/D Ratio: "))
    game_duration_remaining = float(input("Game Duration Remaining (in minutes): "))
    win_rate = float(input("Win Rate (as a decimal, e.g., 0.60 for 60%): "))

    return {
        'mean_opponent_score': mean_opponent_score,
        'current_score_variability': current_score_variability,
        'kd_ratio': kd_ratio,
        'game_duration_remaining': game_duration_remaining,
        'win_rate': win_rate
    }

user_sample = get_user_input()
user_sample_df = pd.DataFrame([user_sample])

winning_probability = mod.predict(user_sample_df)
print(f'\nPredicted Winning Probability for the provided player statistics: {winning_probability[0]:.2f}')














