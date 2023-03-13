import json
import pandas as pd
import streamlit as st
import xgboost as xgb

# Load data
# df = pd.read_csv("C:\Users\KingRemy\OneDrive - University of Keele\Documents\Collaborative App Development\Coursework\Stored_dataset\client_219_153_EDA.csv", header=0, delimiter=',')




def predict_period(Season,Num_of_ticket,StartDate):
    # Loading the saved model
    model_period = xgb.XGBRegressor()
    model_period.load_model("purchase_period_model.json")

    # dict_out = {}

    if Season == 'Autumn':
        Season == 0
    elif Season == 'Winter':
        Season == 3
    elif Season == 'Spring':
        Season == 1
    elif Season == 'Summer':
        Season == 2

    # StartWeek = StartDate.dt.isocalendar().week
    StartHour = StartDate.dt.isocalendar().hour
    StartDayofWeek = StartDate.dt.isocalendar().dayofweek
    StartQuarter = StartDate.dt.isocalendar().quarter
    StartDayofyear = StartDate.dt.isocalendar().dayofyear
    StartMonth = StartDate.dt.isocalendar().month
    StartYear = StartDate.dt.isocalendar().year
    StartDayofMonth = StartDate.dt.isocalendar().day
    StartWeekofYear = StartDate.dt.isocalendar().weekofyear
    prediction_out = model_period.predict(pd.DataFrame([[Season,Num_of_ticket,StartHour,StartDayofWeek,StartQuarter, StartDayofyear, StartMonth, StartYear,StartDayofMonth,StartWeekofYear]], columns=[Season,Num_of_ticket,StartHour,StartDayofWeek,StartQuarter, StartDayofyear, StartMonth, StartYear,StartDayofMonth,StartWeekofYear]))
    # df['Purchase_period_Predicted'] = prediction_out
    return prediction_out