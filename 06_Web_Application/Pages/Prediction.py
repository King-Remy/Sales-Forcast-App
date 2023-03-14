import streamlit as st
import numpy as np
import json
import sys
import pandas as pd

sys.path.insert(1, "06_Web_Application/make_pred.py")

from make_pred import predict_period
# Setup data from csv
# df = pd.read_csv("C:\Users\KingRemy\OneDrive - University of Keele\Documents\Collaborative App Development\Coursework\Stored_dataset\client_219_153_EDA.csv", header=0, delimiter=',')


#Caching the model for faster loading
# @st.cache


# Setup title page
st.set_page_config(page_title="Prediction")
st.header("Prediction - Client Dataset")
st.markdown("Using XGBoost, make predictions on the distribution of likely sales of ticket prior to the start date of an event to decide if more promotion is required to reach target bookings / sales"
            "The prediction will appear on the graphs and table below to intuit how the prediction was made.")
st.sidebar.header("Make Prediction")

start_date = st.sidebar.text_input("Event Start Date")
weeks_to_event = st.sidebar.text_input("Promotion Start (Weeks to Event)")
make_pred = st.sidebar.button("Predict")

# Managing input data
p1 = ""

# Making prediction and displaying data
if make_pred:
    p1 = pd.to_datetime(p1)         # Converting startdate input into datetime
    purchase_period_prediction = predict_period(p1)
    
    st.text_area("Predicted purchase period", value=purchase_period_prediction,height=40)
    # st.subheader(f"Predicted Species: {species_pred}")