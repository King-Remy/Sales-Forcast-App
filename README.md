# ✅ Project Descriptions
<p align="justify"> This project is project is set out to create a model to forecast the likely sales by either event ticket number or value based on sales to date to enable users decide if more promotion to reach a targeted bookings / sales is required. This repository consists of mainly Jupyter Notebook files.

## ✅ Technologies and libraries used
| Name  | Brief |
| ------------- | ------------- |
| Streamlit  |  *An open source app framework for building beautiful data and ML apps* |
| Numpy  | *Python library for numerical computation*  |
| Pandas | *Python library for data manipulation and analysis* |
| JSON | *Python library using JSON API client library*|
| datetime | *Python library using new date and time classes* |
| XGBoost | *Python library using XGBoost Python Package*|
| time | *Python library using time module*|
| itertools | *Python library itertools module*|
| Scipy | *An open-source software for mathematics, science, and engineering*|
| Matplotlib | *Python library for creating static, animated, and interactive visualizations*|
| Seabron | *Python library based on matplotlib*|
| Scikit-learn | *Python library for machine learning*|

## ✅ Python Dependencies Installation
To get started, clone the Git repo onto your laptop using this command on your terminal in your preferred directory:
```
git clone https://github.com/King-Remy/Sales-Forcast-App.git
```

### Folder Structure
| File  | Description |
| ------------- | ------------- |
| /Sales-Forcast-App  |  *Root directory* |
| /01_import/01_import_data.ipynb  | *This notebook merges the 2 data*  |
| /02_Preprocessing/01_Sampling.ipynb | *This script creates a dataset of the sampled dataset and applies some initial cleaning steps* |
| /02_Preprocessing/02_Data_audit.ipynb | *This script takes an audit on the dataset. It looks at how many missing values each feature has, how many unique values the categorical features have then applies pre-processing steps to remove outliers and drop features with too many missing values*|
| /02_Preprocessing/03_corr_and_chi2.ipynb | *This script finds the correlations between numerical features of the dataset and performs a chi-squared test on all dataset variables* |
| /03_EDA/01_EDA.ipynb | *This script performs exploratory data analysis on the dataset*|
| /04_Baseline_Model/01_Baseline_Model.ipynb | *This script applies a linear regression model to get weekly sales distribution, giving a baseline performance for comparison*|
| /05_Modeling/01_Modeling.ipynb | *2 different supervised learning regression models are assessed to understand how well they could predict weekly sales distribution*|
| /06_Web_Application /Dashboard.py | *Creates the dashboard page for the streamlit framework web application*|
| /06_Web_Application /Prediction.py | *Creates the prediction page for the streamlit framework web application*|
| /config-example.json | *JSON file containing elements for loading files*|
| /config-explanation.md | *Explains what each element of the config JSON file refers to*|

Read config-example.json and config-explanation.md to understand the loading of files template.

To install the python dependencies, create a Python virtual environment for the model:

```
python3 -m venv ~/SalesApp
```

Run this command whenever you open a new Terminal window/tab to activate the environment.

```
# Unix/MacOs
source SalesApp/bin/activate
# Windows
.\SalesApp\Scripts\activate
```

Next, to install the python dependencies run the following:
```
# Install Python dependencies
python3 -m pip install pip --upgrade
python3 -m pip install -r requirements.txt
```

Next, run each file from 01_import to 05_Modeling.

## ✅ Web Application Usage

Below demonstrates how the web application is used to obtain sales predictions.

The web application can be found at [here](https://king-remy-sales-forcast-app-06-web-applicationdashboard-8ozcug.streamlit.app/).
![webapp1](/Images/webapp1.png)

1.	User inputs their event type, event start date, booking period (Weeks to event start date) and clicks the predict button.
![webapp2](/Images/webapp2.png)

2.	The formatted inputs are sent to streamlit API and returns all corresponding sales prediction, cumulated sales prediction, cumulated sales percentage, and the weeks to event as a table with the plotted predicted sales Vs weeks to event.
![webapp3](/Images/webapp3.png)
![webapp31](/Images/webapp31.png)

3.	The formatted user inputs are sent to streamlit API and also returns summary of output from the table and steps to take with evaluating performance of event towards the start. 
![webapp4](/Images/webapp4.png)

