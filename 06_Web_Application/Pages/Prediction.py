import streamlit as st
import numpy as np
import json
import datetime
from datetime import date, timedelta
import pandas as pd
import xgboost as xgb
import os


# Setup data from csv
# df = pd.read_csv("C:\Users\KingRemy\OneDrive - University of Keele\Documents\Collaborative App Development\Coursework\Stored_dataset\client_219_153_EDA.csv", header=0, delimiter=',')

# load encoded eventtype file

with open('eventtype_encoder.json') as f:
    config = json.load(f)

#Caching the model for faster loading
@st.cache

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
    prediction_out = model_period.predict(pd.DataFrame([[Season,StartHour,StartDayofWeek,StartQuarter, StartDayofyear, StartMonth, StartYear,StartDayofMonth,StartWeekofYear]], columns=[Season,StartHour,StartDayofWeek,StartQuarter, StartDayofyear, StartMonth, StartYear,StartDayofMonth,StartWeekofYear]))
    # df['Purchase_period_Predicted'] = prediction_out
    return prediction_out

# Setup title page
st.set_page_config(page_title="Prediction")
st.header("Prediction - Client Dataset")
st.markdown("Using XGBoost, make predictions on the distribution of likely sales of ticket prior to the start date of an event to decide if more promotion is required to reach target bookings / sales"
            "The prediction will appear on the graphs and table below to intuit how the prediction was made.")
st.sidebar.header("Make Prediction")

# Creating defaul date for event start date input
today =date.today()

# Creating inputs and button
event_type = st.selectbox("Event Type:", config.keys() )
start_date = st.sidebar.date_input("Event Start Date", today)
weeks_to_event = st.sidebar.number_input("Promotion Start (Weeks to Event)", min_value=0, max_value=100, value=1)
make_pred = st.sidebar.button("Predict")

# Managing input data
p1 = ""

# Making prediction and displaying data
if make_pred:
    p1 = pd.to_datetime(p1)         # Converting startdate input into datetime
    purchase_period_prediction = predict_period(start_date)
    
    st.text_area("Predicted purchase period", value=purchase_period_prediction,height=40)
    # st.subheader(f"Predicted Species: {species_pred}")