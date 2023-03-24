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

def eventTypeConversion(df, event_type, purchase_period):
    
    eventTypeCode = []
    for key,value in config.items():
        if event_type == key: eventTypeCode = list(repeat(value, purchase_period))

    df['EventType'] = eventTypeCode
    return df

def event_startdate_features(StartDate_df):
    StartDate_df = StartDate_df.copy()
    StartDate_df = addSeasonCode(StartDate_df)
    # StartDate_df['StartHour'] = StartDate_df.StartDate.dt.hour
    StartDate_df['StartDayofWeek'] = StartDate_df.StartDate.dt.dayofweek
    StartDate_df['StartQuarter'] = StartDate_df.StartDate.dt.quarter
    StartDate_df['StartDayofyear'] = StartDate_df.StartDate.dt.dayofyear
    StartDate_df['StartMonth'] = StartDate_df.StartDate.dt.month
    StartDate_df['StartYear'] = StartDate_df.StartDate.dt.year
    StartDate_df['StartDayofMonth'] = StartDate_df.StartDate.dt.day
    StartDate_df['StartWeekofYear'] = np.uint32(np.int32(StartDate_df.StartDate.dt.isocalendar().week))
    StartDate_df['StartDate'] = StartDate_df.StartDate.dt.date
    return StartDate_df

def booking_startdate_feautre(booking_dates_df):
    booking_dates_df = booking_dates_df.copy()
    # booking_dates_df['BookedHour'] = booking_dates_df.StatusCreatedDate.dt.hour
    
    booking_dates_df['StatusCreatedDayofWeek'] = booking_dates_df.StatusCreatedDate.dt.dayofweek
    booking_dates_df['StatusCreatedQuarter'] = booking_dates_df.StatusCreatedDate.dt.quarter
    booking_dates_df['StatusCreatedDayofyear'] = booking_dates_df.StatusCreatedDate.dt.dayofyear
    booking_dates_df['StatusCreatedMonth'] = booking_dates_df.StatusCreatedDate.dt.month
    booking_dates_df['StatusCreatedYear'] = booking_dates_df.StatusCreatedDate.dt.year
    booking_dates_df['StatusCreatedDayofMonth'] = booking_dates_df.StatusCreatedDate.dt.day
    booking_dates_df['StatusCreatedWeekofYear'] = booking_dates_df.StatusCreatedDate.dt.weekofyear
    booking_dates_df['StatusCreatedDate'] = booking_dates_df.StatusCreatedDate.dt.date

    return booking_dates_df


def predict_period(StartDate):          # This function takes in a DatFrame with Event StartDate to break down its features and predict purchase period 
    df2 = event_startdate_features(Client).drop(labels=['StartDate', 'StartSeason'], axis=1)
    period_pred_out = model_period.predict(df2)
    # df['Purchase_period_Predicted'] = prediction_out
    return round(period_pred_out[0])

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
start_time = dt.datetime.strptime('0000','%H%M').time()

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
    # conv = str(start_datetime)
    # p1 = pd.to_datetime(start_datetime, utc=True)  
    Client = pd.DataFrame.from_dict([{"StartDate": start_date}])
    Client["StartDate"] = pd.to_datetime(start_date,errors='coerce')              # converting created Event Startdate column with users StartDate to datetime format

    # purchase_period_prediction = predict_period(Client)
    
    sales_weeks_df = pd.DataFrame(ticket_sales_features(pd.to_datetime(start_date,errors='coerce'), weeks_to_event,event_type))
    sales_weeks_pred = predictWeeklySales(sales_weeks_df)

    # st.success(f"Predicted purchase period {purchase_period_prediction}")

    # st.success(f"Predicted purchase period {sales_weeks_pred}")
    # st.dataframe(sales_weeks_pred, use_container_width=True)
    # st.subheader(f"Predicted Species: {species_pred}")
    totalSales = sales_weeks_pred['Sales Prediction'].sum()
    st.success(f"Based on your preferred event type and booking period for the weeks to your event start date, the total number of ticket sales for your event is {str(totalSales)}", icon="ℹ️")
    
    st.info("Weekly sales distribution predictions")
    st.dataframe(sales_weeks_pred, use_container_width=True)
    st.info("Plotting Week vs Sales Prediction")
    st.line_chart(sales_weeks_pred, x='Weeks to Event', y='Sales Prediction')

    st.success("Summary")

    st.markdown(f"From the predictions above, it shows the sales distribution to your preferred booking period of {weeks_to_event} (weeks to event start date). The table above displays the weeks to event and the week starting date of the year, the predicted sales of that week and the cummulated Sales prediction and booking percentage to the week of event start date.")
    st.markdown("Below shows the checkpoint stages based on percentiles (25%, 50%, and 90%) of the cumlated sales predicted column")
    st.subheader(f"Checkpoint 1 (25th percentile) - {round(sales_weeks_pred['Cummulated Sales Prediction'].quantile(0.25))} tickets by {round(sales_weeks_pred['Weeks to Event'].quantile(0.25))} weeks to event")
    st.subheader(f"Checkpoint 2 (50th percentile) - {round(sales_weeks_pred['Cummulated Sales Prediction'].quantile(0.5))} tickets by {round(sales_weeks_pred['Weeks to Event'].quantile(0.5))} weeks to event")
    st.subheader(f"Checkpoint 3 (90th percentile) - {round(sales_weeks_pred['Cummulated Sales Prediction'].quantile(0.9))} tickets by {round(sales_weeks_pred['Weeks to Event'].quantile(0.9))} weeks to event")

    st.markdown("By matching your actual cumulated sales by the predicted cumulated sales, if the actual is higher/below the predicted, please consider increasing/reducing the size of venue or the promotions.")
                
    # sales_table = pd.DataFrame()

    # Creating weeks to event date column
    # date_plus_weeks_added = start_date + timedelta(weeks=weeks_to_event)

    # value = range(weeks_to_event)

    # sales_table['Weeks to Event (Date)'] = pd.date_range(start=date_plus_weeks_added, end=start_datetime, freq='W')

    # sales_table['Weeks to Event (Number)'] = weeks_to_event

    # for i in value:
    #     purchase_sales prediction = predict_sales(p1, event_type, weeks_to_event)

        
    #     sales_table['Weeks to Event (Number)'] = sales_table['Weeks to Event (Number)'] - 1
    #     # sales_table['Weeks to Event (Number)'] = sales_table['Weeks to Event (Number)'].apply(lambda x: x -1)
    #     sales_table['Number of Tickets (predicted)'] = purchase_sales prediction.tolist
    #     sales_table['Sales (Cum_Sum)'] = sales_table['Number of Tickets (predicted)'].cumsum()
    #     sales_table['Sales (Cum_Perc)'] = 100*sales_table['Sales (Cum_Sum)']/sales_table['Number of Tickets (predicted)'].sum()

    # st.dataframe(sales_table)