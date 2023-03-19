import streamlit as st
import numpy as np
import json
import datetime
from datetime import date, timedelta
import pandas as pd
import xgboost as xgb
import os
import time


# Setup data from csv
# df = pd.read_csv("C:\Users\KingRemy\OneDrive - University of Keele\Documents\Collaborative App Development\Coursework\Stored_dataset\client_219_153_EDA.csv", header=0, delimiter=',')

# load encoded eventtype file
config_path = "/app/sales-forcast-app/06_Web_Application/Pages"

with open(config_path + '/eventtype_encoder.json', 'r') as f:
    config = json.load(f)

# Loading the saved model
model_period = xgb.XGBRegressor()
model_period.load_model("06_Web_Application/Pages/purchase_period_model.json")

model_sales = xgb.XGBRegressor()
model_sales.load_model("06_Web_Application/Pages/weekly_sales_model.json")

#Caching the model for faster loading
@st.cache_resource

def addSeasonCode(df):
    # Creating the Season column
    _condition_winter = (df.StartDate.dt.month>=1)&(df.StartDate.dt.month<=3)
    _condtion_spring = (df.StartDate.dt.month>=4)&(df.StartDate.dt.month<=6)
    _condition_summer = (df.StartDate.dt.month>=7)&(df.StartDate.dt.month<=9)
    _condition_autumn = (df.StartDate.dt.month>=10)&(df.StartDate.dt.month<=12)
    
    df['StartSeason'] = np.where(_condition_winter,'Winter',np.where(_condtion_spring,'Spring',np.where(_condition_summer,'Summer',np.where(_condition_autumn,'Autumn',np.nan))))

    eventSeasonCode = []
    for row in df['StartSeason']:
        if row == 'Autumn': eventSeasonCode.append(0)
        if row == 'Winter': eventSeasonCode.append(3)
        if row == 'Spring': eventSeasonCode.append(1)
        if row == 'Summer': eventSeasonCode.append(2)

    df['Season'] = eventSeasonCode

    return df

def event_startdate_features(StartDate_df):
    StartDate_df = StartDate_df.copy()
    StartDate_df = addSeasonCode(StartDate_df)
    StartDate_df['StartHour'] = StartDate_df.dt.hour
    StartDate_df['StartDayofWeek'] = StartDate_df.dt.dayofweek
    StartDate_df['StartQuarter'] = StartDate_df.dt.quarter
    StartDate_df['StartDayofyear'] = StartDate_df.dt.dayofyear
    StartDate_df['StartMonth'] = StartDate_df.dt.month
    StartDate_df['StartYear'] = StartDate_df.dt.year
    StartDate_df['StartDayofMonth'] = StartDate_df.dt.day
    StartDate_df['StartWeekofYear'] = StartDate_df.dt.weekofyear
    StartDate_df['StartDate'] = StartDate_df.dt.date
    return StartDate_df

def predict_period(StartDate):          # This function takes in a DatFrame with Event StartDate to break down its features and predict purchase period 
    df2 = event_startdate_features(Client).drop(labels=['StartDate', 'StartSeason'], axis=1)
    period_pred_out = model_period.predict(df2)
    # df['Purchase_period_Predicted'] = prediction_out
    return round(period_pred_out[0])

def predict_sales(StartDate, event_type, weeks_to_event):
    StatusCreatedHour = StartDate.hour
    StatusCreatedDayofWeek = StartDate.dayofweek
    StatusCreatedQuarter = StartDate.quarter
    StatusCreatedDayofyear = StartDate.dayofyear
    StatusCreatedMonth = StartDate.month
    StatusCreatedYear = StartDate.year
    StatusCreatedDayofMonth = StartDate.day
    StatusCreatedWeekofYear = StartDate.weekofyear

    for key,value in config.items():
        if event_type == key:
            event_type = value

    prediction_sales_out = model_sales.predict(pd.DataFrame([[StatusCreatedHour,event_type,StatusCreatedDayofWeek,StatusCreatedQuarter,StatusCreatedDayofyear,StatusCreatedMonth,StatusCreatedYear,StatusCreatedDayofMonth,StatusCreatedWeekofYear,weeks_to_event]], columns=['StatusCreatedHour','StartType','StatusCreatedDayofWeek','StatusCreatedQuarter','StatusCreatedDayofyear','StatusCreatedMonth','StatusCreatedYear','StatusCreatedDayofMonth','StatusCreatedWeekofYear','Weeks to Event']))

    return prediction_sales_out

# Setup title page
st.set_page_config(page_title="Prediction")
st.header("Prediction - Client Dataset")
st.markdown("Using XGBoost, make predictions on the distribution of likely sales of ticket prior to the start date of an event to decide if more promotion is required to reach target bookings / sales"
            "The prediction will appear on the graphs and table below to intuit how the prediction was made.")
st.sidebar.header("Make Prediction")


# Creating inputs and button
event_type = st.sidebar.selectbox("Event Type:", config.keys() )
start_date = st.sidebar.date_input("Event Start Date", datetime.date.today())
start_time = st.sidebar.time_input('Enter start time', datetime.time(0, 00))

start_datetime = datetime.datetime.combine(start_date, start_time)

weeks_to_event = st.sidebar.number_input("Booking Period (Weeks to EventDate)", min_value=0, max_value=100, value=1)
make_pred = st.sidebar.button("Predict")

# Managing input data
# p1 = ""

# Making prediction and displaying data
if make_pred:
    with st.spinner('Generating predictions. Please wait....'):
        time.sleep(1)
    
    

    # Generating data frame
    conv = str(start_datetime)
    Client = pd.DataFrame.from_dict([{"StartDate": conv}])
    Client["StartDate"] = pd.to_datetime(Client["StartDate"], infer_datetime_format=True)                # converting created Event Startdate column with users StartDate to datetime format

    purchase_period_prediction = predict_period(Client)
    
    st.success(f"Predicted purchase period {purchase_period_prediction}")
    # st.subheader(f"Predicted Species: {species_pred}")

    

    # sales_table = pd.DataFrame()

    # Creating weeks to event date column
    # date_plus_weeks_added = start_date + timedelta(weeks=weeks_to_event)

    # value = range(weeks_to_event)

    # sales_table['Weeks to Event (Date)'] = pd.date_range(start=date_plus_weeks_added, end=start_datetime, freq='W')

    # sales_table['Weeks to Event (Number)'] = weeks_to_event

    # for i in value:
    #     purchase_sales_prediction = predict_sales(p1, event_type, weeks_to_event)

        
    #     sales_table['Weeks to Event (Number)'] = sales_table['Weeks to Event (Number)'] - 1
    #     # sales_table['Weeks to Event (Number)'] = sales_table['Weeks to Event (Number)'].apply(lambda x: x -1)
    #     sales_table['Number of Tickets (predicted)'] = purchase_sales_prediction.tolist
    #     sales_table['Sales (Cum_Sum)'] = sales_table['Number of Tickets (predicted)'].cumsum()
    #     sales_table['Sales (Cum_Perc)'] = 100*sales_table['Sales (Cum_Sum)']/sales_table['Number of Tickets (predicted)'].sum()

    # st.dataframe(sales_table)
