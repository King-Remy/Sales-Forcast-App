import streamlit as st
import numpy as np
import json
import datetime
import datetime as dt
import pandas as pd
import xgboost as xgb
import os
import time
from itertools import repeat

# Setup data from csv
# df = pd.read_csv("C:\Users\KingRemy\OneDrive - University of Keele\Documents\Collaborative App Development\Coursework\Stored_dataset\client_219_153_EDA.csv", header=0, delimiter=',')

# load encoded eventtype file
config_path = "/app/sales-forcast-app/06_Web_Application/pages"

with open(config_path + '/eventtype_encoder.json', 'r') as f:
    config = json.load(f)

# Loading the saved model
model_period = xgb.XGBRegressor()
model_period.load_model("06_Web_Application/pages/purchase_period_model.json")

model_sales = xgb.XGBRegressor()
model_sales.load_model("06_Web_Application/pages/weekly_sales_model.json")
#Caching the model for faster loading
@st.cache_resource

def eventTypeConversion(df, event_type, purchase_period):           # This function converts the users event type and maps it to the decoded number for the ML model
    
    eventTypeCode = []
    for key,value in config.items():
        if event_type == key: eventTypeCode = list(repeat(value, purchase_period))

    df['EventType'] = eventTypeCode
    return df

def booking_startdate_feautre(booking_dates_df):            # This functions takes in the generated weeks to event booking dates and creates its features
    booking_dates_df = booking_dates_df.copy()
    
    booking_dates_df['StatusCreatedDayofWeek'] = booking_dates_df.StatusCreatedDate.dt.dayofweek
    booking_dates_df['StatusCreatedQuarter'] = booking_dates_df.StatusCreatedDate.dt.quarter
    booking_dates_df['StatusCreatedDayofyear'] = booking_dates_df.StatusCreatedDate.dt.dayofyear
    booking_dates_df['StatusCreatedMonth'] = booking_dates_df.StatusCreatedDate.dt.month
    booking_dates_df['StatusCreatedYear'] = booking_dates_df.StatusCreatedDate.dt.year
    booking_dates_df['StatusCreatedDayofMonth'] = booking_dates_df.StatusCreatedDate.dt.day
    booking_dates_df['StatusCreatedWeekofYear'] = booking_dates_df.StatusCreatedDate.dt.weekofyear
    booking_dates_df['StatusCreatedDate'] = booking_dates_df.StatusCreatedDate.dt.date

    return booking_dates_df

def ticket_sales_features(StartDate, purchase_period, event_type):          # This function creates features required to predict weekly sales distribution
    freq = '-1W-SUN'
    weeks = list(reversed(range(purchase_period)))

    period = pd.date_range(StartDate, periods=purchase_period, freq=freq)
    period = pd.DataFrame(reversed(period))
    period['StartDate'] = StartDate
    period.columns =['StatusCreatedDate', 'StartDate']
    period = eventTypeConversion(period, event_type, purchase_period)
    period = booking_startdate_feautre(period)
    period['Weeks to Event'] = weeks
    return period

    

def predictWeeklySales(df):
    df2 = df.drop(labels=['StatusCreatedDate', 'StartDate'], axis=1)
    weekly_sales_pred_out = model_sales.predict(df2)
    weekly_sales_pred = pd.DataFrame()
    weekly_sales_pred['Weeks to Event'] = df['Weeks to Event']
    weekly_sales_pred['Week Starting'] = pd.to_datetime(df['StatusCreatedDate'])

    predictions = []
    # index = len(weekly_sales_pred_out) + 1
    for row in weekly_sales_pred_out:
        if row < 0:
            predictions.append(abs(round(row)))
        else:
            predictions.append(abs(round(row)))
    
    weekly_sales_pred['Sales Prediction'] = predictions
    weekly_sales_pred['Cummulated Sales Prediction'] = pd.Series(predictions).cumsum()
    weekly_sales_pred['Cummulated Sales %'] = round((weekly_sales_pred['Cummulated Sales Prediction'] / weekly_sales_pred['Sales Prediction'].sum()) * 100, 0)

    return weekly_sales_pred

# Setup title page
st.set_page_config(page_title="Prediction")
st.header("Prediction - Client Dataset")
st.markdown("Using XGBoost, make predictions on the distribution of likely sales of ticket prior to the start date of your event to decide if more promotion is required to reach target bookings / sales"
            "The prediction will appear on the graphs and table below to intuit how the prediction was made.")
st.sidebar.header("Make Prediction")


# Creating inputs and button
event_type = st.sidebar.selectbox("Event Type:", config.keys() )
start_date = st.sidebar.date_input("Event Start Date", datetime.date(2023, 4, 1))

weeks_to_event = st.sidebar.number_input("Booking Period (Weeks to EventDate)", min_value=0, max_value=100, value=1)
make_pred = st.sidebar.button("Predict")

# Making prediction and displaying data
if make_pred:
    with st.spinner('Generating predictions. Please wait....'):
        time.sleep(1)
    
    # Creates dataframe with the start date user input to extract time formats
    Client = pd.DataFrame.from_dict([{"StartDate": start_date}])
    Client["StartDate"] = pd.to_datetime(start_date,errors='coerce')    # converting created Event Startdate column to datetime format
    

    # Creates dataframe with start date, booking period and event type input from user to generate features for predicting weekly sales
    sales_weeks_df = pd.DataFrame(ticket_sales_features(pd.to_datetime(start_date,errors='coerce'), weeks_to_event,event_type))
    # Makes predictions with the generated features used to predict weekly sales
    sales_weeks_pred = predictWeeklySales(sales_weeks_df)

    totalSales = sales_weeks_pred['Sales Prediction'].sum()
    st.success(f"Based on your preferred event type and booking period for the weeks to your event start date, the total number of ticket sales for your event is {str(totalSales)}",icon="ℹ️")
    
    # Displays final dataframe showing weekly sales and corresponding features  
    st.info("Weekly sales distribution predictions")
    st.dataframe(sales_weeks_pred, use_container_width=True)
    # Plots a line graph showing the weekly sales Vs Weeks to Event start date
    st.info("Plotting Week vs Sales Prediction")
    st.line_chart(sales_weeks_pred, x='Weeks to Event', y='Sales Prediction')

    st.success("Summary")

    st.markdown(f"From the predictions above, it shows the sales distribution to your preferred booking period of {weeks_to_event} (weeks to event start date). The table above displays the weeks to event and the week starting date of the year, the predicted sales of that week, the cummulated Sales prediction and booking percentage to the week of event start date.")
    st.markdown("Below shows the checkpoint stages based on percentiles (25%, 50%, and 90%) of the cumlated sales predicted column")
    st.subheader(f"Checkpoint 1 (25th percentile) - {round(sales_weeks_pred['Cummulated Sales Prediction'].quantile(0.25))} tickets by {round(sales_weeks_pred['Weeks to Event'].quantile(0.75))} weeks to event")
    st.subheader(f"Checkpoint 2 (50th percentile) - {round(sales_weeks_pred['Cummulated Sales Prediction'].quantile(0.5))} tickets by {round(sales_weeks_pred['Weeks to Event'].quantile(0.5))} weeks to event")
    st.subheader(f"Checkpoint 3 (90th percentile) - {round(sales_weeks_pred['Cummulated Sales Prediction'].quantile(0.9))} tickets by {round(sales_weeks_pred['Weeks to Event'].quantile(0.1))} weeks to event")

    st.markdown("By matching your actual cumulated sales by the predicted cumulated sales by each checkpoint, if the actual is higher/below the predicted, please consider increasing/reducing the size of venue or the promotions.")
