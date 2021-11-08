from altair import Chart, X, Y, Axis, Data, DataFormat
import pandas as pd
import numpy as np
from flask import render_template, url_for, flash, redirect, request, make_response, jsonify, abort
from web import app
from web.utils import utils, home_plot, macro_plot, market_plot, fedfundfutures_plot
from web.models.fff_model.main import Main

import json
from werkzeug.utils import secure_filename
import os

# Setting data directory: saved to web/data
uploads_dir = os.path.join(os.path.dirname(app.instance_path), 'web/data')
os.makedirs(uploads_dir, exist_ok=True)

main = Main()


# Loading raw data and clean it

#loading and clean data
#market
market_data = utils.load_market_data()
market_data_cleaned = utils.clean_market(market_data)


market_ngram_statement, market_ngram_min = utils.load_ngram_market_data(year=2004)
#market_ngram_min, market_ngram_statement, market_ngram_news = utils.load_ngram_market_data()
#market_ngram_min_cleaned = utils.clean_ngram_data(market_ngram_min)
#market_ngram_statement_cleaned = utils.clean_ngram_data(market_ngram_statement)

#macro
gdp_data, employment_data, inflation_data = utils.load_macro_data()
macro_ts_train, macro_ts_test, macro_X_train, macro_X_test = utils.load_macro_ts()
macro_ts = utils.clean_macro_ts(macro_ts_train, macro_ts_test)
macro_main_data = utils.load_macro_model_data()

macro_maindashboard_data = utils.clean_maindashboard_macro(macro_ts_train, macro_ts_test, macro_X_train, macro_X_test )

#fedfundfuture
fff_data, fake_data = utils.load_fff_data()
fff_data_cleaned = utils.clean_fff(fff_data)

#home
home_data = utils.load_home_data()

#to change the data for home and start page
@app.route("/")
def plot_main_dashboard():
    # actual values to be filled up 
    market_consensus = 1
    macroeconomic_indicators = -1
    fedfundfutures= 1
    #ploting
    #market
    plot_market_senti_main = home_plot.plot_market(market_data)
    plot_market_average = home_plot.plot_market_average(market_data)
    market_sentiments_drill1 = market_plot.display_market_sentiments_drill_down_1(market_data_cleaned)
    
    
    #macro
    plot_gdp_index = home_plot.plot_gdp_index(home_data)
    
    #fff
    plot_fff = home_plot.plot_fff(fff_data)

    # gauge
    plot_gauge = home_plot.plot_gauge()

    context = {"market_consensus": market_consensus,
               "macroeconomic_indicators": macroeconomic_indicators,
               "fedfundfutures": fedfundfutures,
               "plot_market_senti_main": plot_market_senti_main,
               "plot_market_average": plot_market_average,
               'plot_gdp_index': plot_gdp_index, 
               'plot_fff': plot_fff,
               'market_sentiments_drill1': market_sentiments_drill1,
               "plot_gauge": plot_gauge}
    return render_template('home.html', context=context)

@app.route("/upload")
def upload_file():
    return render_template('model-run.html')

@app.route("/runfff", methods = ['GET', 'POST'])
def run_fff_model():
    main.run_main()
    return render_template('model-run.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_files():
   if request.method == 'POST':
      f = request.files['file']
      path = os.path.join(uploads_dir, secure_filename(f.filename))
      f.save(path)
      return 'file uploaded successfully'


@app.route("/home")
def plot_home():
    # actual values to be filled up 
    market_consensus = 1
    macroeconomic_indicators = -1
    fedfundfutures= 1
    #ploting
    #market
    plot_market_senti_main = home_plot.plot_market(market_data, date='2004-09-30')
    plot_market_average = home_plot.plot_market_average(market_data_cleaned, date='2004-09-30')
    market_sentiments_drill1 = home_plot.display_market_sentiments_drill_down_4(market_data_cleaned)
    
    
    #macro
    plot_gdp_index = home_plot.plot_gdp_index(home_data)
    #plot_macro_average = home_plot.plot_market_average2(market_data)
    macro_ts_plot = home_plot.plot_fed_rates_ts(macro_ts)
    print(macro_maindashboard_data.head())
    macro_maindashboard_plot = home_plot.plot_macro_maindashboard(macro_maindashboard_data, date='2004-09-30')
    macro_pie_chart = home_plot.plot_contributions_pie(macro_maindashboard_data, date='2004-09-30')
    #fff
    plot_fff = home_plot.plot_fff(fff_data)
    #plot_fff_average = home_plot.plot_market_average3(market_data)

    # gauge
    plot_gauge = home_plot.plot_gauge()

    context = {"market_consensus": market_consensus,
               "macroeconomic_indicators": macroeconomic_indicators,
               "fedfundfutures": fedfundfutures,
               "plot_market_senti_main": plot_market_senti_main,
               "plot_market_average": plot_market_average,
               #"plot_macro_average": plot_macro_average,
               #"plot_fff_average": plot_fff_average,
               'plot_gdp_index': plot_gdp_index, 
               'plot_fff': plot_fff,
               'market_sentiments_drill1': market_sentiments_drill1,
               "plot_gauge": plot_gauge,
               'macro_ts_plot':macro_ts_plot,
               'macro_maindashboard_plot':macro_maindashboard_plot,
               'macro_pie_chart':macro_pie_chart}
    return render_template('home.html', context=context)


@app.route("/market-consensus")
def plot_market_consensus():
    #ploting
    market_sentiments_drill1 = market_plot.display_market_sentiments_drill_down_1(market_data_cleaned)
    market_sentiments_drill2 = market_plot.display_market_sentiments_drill_down_2(market_data_cleaned)
    market_sentiments_drill3 = market_plot.display_market_sentiments_drill_down_3(market_data_cleaned)
    market_ts_plot = market_plot.plot_hd_ts(market_data_cleaned)
    ngram_min = market_plot.get_top_n_gram_mins(market_ngram_min)
    ngram_statement = market_plot.get_top_n_gram_st(market_ngram_statement)

    context = {'market_sentiments_drill1': market_sentiments_drill1,
               'market_sentiments_drill2': market_sentiments_drill2,
               'market_sentiments_drill3': market_sentiments_drill3, 
               'market_ts_plot':market_ts_plot,
               'ngram_min': ngram_min, 
               'ngram_statement':ngram_statement}
    return render_template('market-consensus.html', context=context)


@app.route("/macroeconomic-indicators")
def plot_macroeconomic_indicators():
    #ploting
    plot_gdp_index = macro_plot.plot_gdp_index(gdp_data)
    plot_employment_index = macro_plot.plot_employment_index(employment_data)
    plot_inflation_index = macro_plot.plot_inflation_index(inflation_data)
    plot_main_model = macro_plot.plot_main_plot(macro_main_data)
    plot_indicators_ts = macro_plot.plot_indicators_ts(macro_maindashboard_data ) 
    context = {'plot_gdp_index': plot_gdp_index, 
               'plot_employment_index': plot_employment_index, 
               'plot_inflation_index': plot_inflation_index, 
               'plot_main_model': plot_main_model, 
               'plot_indicators_ts':plot_indicators_ts}
    return render_template('macroeconomic-indicators.html', context=context)

    
@app.route("/fedfundfutures")
def plot_fedfundfutures():
    #ploting - add plots here and in context
    plot_fff_results = fedfundfutures_plot.plot_fff_results(fff_data)
    context = {'plot_fff_results': plot_fff_results} 
    return render_template('fedfundfutures.html', context=context)

