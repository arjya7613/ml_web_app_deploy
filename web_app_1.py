# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 23:01:04 2024

@author: HP
"""


import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

mod = pickle.load(open('C:\\Users\\HP\\OneDrive\\Desktop\\hackathon\\model.sav','rb'))


#create function for prediction
def Winning_Probability(input_data):

    df=pd.read_csv('C:\\Users\\HP\\OneDrive\\Desktop\\hackathon\\player_performance_custom_data.csv')
    
    #Convert the input data to a DataFrame
    input_df = pd.DataFrame([input_data], columns=['mean_opponent_score', 'current_score_variability', 'kd_ratio', 'game_duration_remaining', 'win_rate'])
       
       # Predict using the loaded model
    prediction = mod.predict(input_df)
       
    return prediction[0]



def main():
    #title for web page 
    st.title('Winning Probability')
    
    # getting input data from user
    
    mean_opponent_score = st.text_input('Mean Opponent Score: ')
    current_score_variability = st.text_input('Current Score Variability: ')
    kd_ratio = st.text_input('K/D Ratio: ') 
    game_duration_remaining = st.text_input('Game Duration Remaining (in minutes): ') 
    win_rate = st.text_input('Win Rate (as a decimal, e.g., 0.60 for 60%): ') 
    

    analyse = " "
    
    #create a button
    if st.button('Prediction'):
   
        try:
            # Convert inputs to float and pass to model
            input_data = [
                float(mean_opponent_score), 
                float(current_score_variability), 
                float(kd_ratio), 
                float(game_duration_remaining), 
                float(win_rate)
            ]
            prediction = Winning_Probability(input_data)
            analyse = f'Predicted Winning Probability: {prediction:.2f}'
        except ValueError:
            analyse = "Please enter valid numeric values for all fields."
            
    st.success(analyse)
    
if __name__ == '__main__':
    main()