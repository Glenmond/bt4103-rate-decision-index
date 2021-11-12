# Rate Decision Indicators

## Table of Contents
1. Project Description
2. Installation
3. Running Instructions
4. Data Understanding
5. Code Description

## 1. Project Description
Our project aims to create a holistic and representative Central Bank Rate Decision Indicator leveraging on economics, finance, statistical
 knowledge to project the next rate decision by the US Federal Reserve.
#### Objectives/Goals
1. Develop a robust system and pipeline to extract economic and financial data from various data sources.
2. Generate statistical, seasonality and sentiment insights from the data for model signals to
facilitate discretionary trade decisions.
3. Create predictive mathematical models, coupled with machine learning and natural
language processing models to forecast central bank rate decisions.
4. Construct interactive dashboards and visualisations to display useful performance metrics
and business insights that fulfil business users needs.
5. Create robust forecasting analysis of data to identify and construct adverse scenarios that
are naturally interpretable for stress testing purposes.
#### Overview
The Central Bank Rate Decision Indicator consist of three sub-models, namely Macroeconomic Model, Market Consensus
 Model and Fed Funds Futures Probability Model. Here is a brief summary for each sub-models:
 1. Macroeconomic Model
 
This model utilises 4 macroeconomic indicators (inflation, economic output, employment and bond yield spread) to predict the federal funds rate. To replicate the impact of economic data on the Central Bank rate decision making process, we employed data processing methods such as lead-lag analysis to account for leading and lagging effects of economic data versus the actual state of the economy. A regression model was then built to obtain a prediction of the Federal Funds Rate given these economic indicators. 

 
 2. Market Consensus Model
 
 This model utilises the studying of FOMC Documents (Statements & Minutes), as well as news articles (New York Times
 ) to generate a Hawkish-Dovish index to represent the markets sentiments. We utilise a heuristic dictionary-based
  approach, as well as machine learning models to train and test the model on a labelled dataset. (+1: Hawkish, -1
  : Dovish).
 

 3. Fed Funds Futures Probability Model
 
The primary objective of the Federal Funds Futures model is to capture the market expectation of future federal funds rate and to calculate a probability for the rate change for each future FOMC meeting.

## 2. Installation
#### Libraries
Required libraries are described in requirements.txt. The code should run with no issues using Python versions 3.7+.
There are 2 ways that you can create the virtual environment required:

1. Using the requirements.txt file using Anaconda
```
conda create -n rate python=3.7 jupyter
conda activate rate
pip install -r requirements.txt
```

2. Using the environments.yml file
```
conda env create -f environment.yml
conda activate rate
```

To verify that all packages are installed correctly, you can run: 
```
conda env list
```

## 3. Running Instructions
#### To Run Web Applications
1. cd to the rate_decision_index directory, type in the command prompt:
```
cd rate_decision_index
```

2. Run the web application, type in the command prompt:
```
python run.py
```

#### To Run Notebook (Local)
#### 3a) Macroeconomics Model

1. Move to models
```
cd rate_decision_index/models
```
2. To download and update all data
```
python update_macro_data.py
```

3. To run the model
```
python macro_analysis.py
```

#### 3b) Sentiment Model
##### Step 1: Download input data
1. Move to extract directory
   ```
   cd rate_decision_index/models/extract
   ```
2. Get data from FOMC Website and New York Times. Specify document type. You can also specify from year.
   ```
   python import_data.py all 2021
   ```
    Note: You can specify and change the from_year date. (ie, type `python import_data.py all 2010` for data from
    year 2010 onwards.)
##### Step 2: Run sentiment models
1. Move to sentiment_model directory
   ```
   cd ../sentiment_model
   ```
2. Get data from FOMC Website. Specify document type. You can also specify from year.
   ```
   python main.py 2021
   ```
   Note: You can specify and change the from_year date. (ie, type `python main.py 2010` for data from
   year 2010 onwards.)
##### Extra: Model Development (Run on Jupyter notebooks for training purposes)
1. Go to top directory
   `cd notebooks`
2. Run the jupyter notebooks 
   `jupyter notebook`
3. Open and run notebooks No.1 to No.4 for analysis

#### 3c) Feds Futures Probability Model
 
1. Move to the fed_futures_model
```
cd rate_decision_index/models/fed_futures_model
```

2. To run the model (and concurrently download and update all data)
```
python main.py
```
