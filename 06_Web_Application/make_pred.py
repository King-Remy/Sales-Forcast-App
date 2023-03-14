import json
import pandas as pd
import numpy as np
import streamlit as st
import xgboost as xgb

# Load data
# df = pd.read_csv("C:\Users\KingRemy\OneDrive - University of Keele\Documents\Collaborative App Development\Coursework\Stored_dataset\client_219_153_EDA.csv", header=0, delimiter=',')




def predict_period(StartDate):
    # Loading the saved model
    model_period = xgb.XGBRegressor()
    model_period.load_model("purchase_period_model.json")

    # dict_out = {}
    
    # Creating the Season column
    _condition_winter = (StartDate.month>=1)&(StartDate.month<=3)
    _condtion_spring = (StartDate.month>=4)&(StartDate.month<=6)
    _condition_summer = (StartDate.month>=7)&(StartDate.month<=9)
    _condition_autumn = (StartDate.month>=10)&(StartDate.month<=12)
    Season = np.where(_condition_winter,'Winter',np.where(_condtion_spring,'Spring',np.where(_condition_summer,'Summer',np.where(_condition_autumn,'Autumn',np.nan))))

    if Season == 'Autumn':
        Season = 0 
    elif Season == 'Winter':
        Season = 3
    elif Season == 'Spring':
        Season = 1 
    elif Season == 'Summer':
        Season = 2

    # StartWeek = StartDate.week
    StartHour = StartDate.hour
    StartDayofWeek = StartDate.dayofweek
    StartQuarter = StartDate.quarter
    StartDayofyear = StartDate.dayofyear
    StartMonth = StartDate.month
    StartYear = StartDate.year
    StartDayofMonth = StartDate.day
    StartWeekofYear = StartDate.weekofyear
    prediction_out = model_period.predict(pd.DataFrame([[Season,StartHour,StartDayofWeek,StartQuarter, StartDayofyear, StartMonth, StartYear,StartDayofMonth,StartWeekofYear]], columns=[Season,Num_of_ticket,StartHour,StartDayofWeek,StartQuarter, StartDayofyear, StartMonth, StartYear,StartDayofMonth,StartWeekofYear]))
    # df['Purchase_period_Predicted'] = prediction_out
    return prediction_out

# def predict_sales(StartDate, EventType, Weeks_to_event):
#     if EventType == 'Colloquium':
        
