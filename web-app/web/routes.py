from altair import Chart, X, Y, Axis, Data, DataFormat
import pandas as pd
import numpy as np
from flask import render_template, url_for, flash, redirect, request, make_response, jsonify, abort
from web import app
from web.utils import utils, home_plot, macro_plot, market_plot
import json

# Loading raw data and clean it

#loading and clean data
#market
market_data = utils.load_market_data()
market_data_cleaned = utils.clean_market(market_data)

#macro
gdp_data = utils.load_macro_data()

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

    context = {"market_consensus": market_consensus,
               "macroeconomic_indicators": macroeconomic_indicators,
               "fedfundfutures": fedfundfutures,
               "plot_market_senti_main": plot_market_senti_main,
               "plot_market_average": plot_market_average,
               'plot_gdp_index': plot_gdp_index, 
               'plot_fff': plot_fff,
               'market_sentiments_drill1': market_sentiments_drill1}
    return render_template('home.html', context=context)

@app.route("/home")
def plot_home():
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

    context = {"market_consensus": market_consensus,
               "macroeconomic_indicators": macroeconomic_indicators,
               "fedfundfutures": fedfundfutures,
               "plot_market_senti_main": plot_market_senti_main,
               "plot_market_average": plot_market_average,
               'plot_gdp_index': plot_gdp_index, 
               'plot_fff': plot_fff,
               'market_sentiments_drill1': market_sentiments_drill1}
    return render_template('home.html', context=context)


@app.route("/market-consensus")
def plot_market_consensus():
    #ploting
    market_sentiments_drill1 = market_plot.display_market_sentiments_drill_down_1(market_data_cleaned)
    market_sentiments_drill2 = market_plot.display_market_sentiments_drill_down_2(market_data_cleaned)
    market_sentiments_drill3 = market_plot.display_market_sentiments_drill_down_3(market_data_cleaned)
    context = {'market_sentiments_drill1': market_sentiments_drill1,
               'market_sentiments_drill2': market_sentiments_drill2,
               'market_sentiments_drill3': market_sentiments_drill3}
    return render_template('market-consensus.html', context=context)


@app.route("/macroeconomic-indicators")
def plot_macroeconomic_indicators():
    #ploting
    plot_gdp_index = macro_plot.plot_gdp_index(gdp_data)
    context = {'plot_gdp_index': plot_gdp_index}
    return render_template('macroeconomic-indicators.html', context=context)

    
@app.route("/fedfundfutures")
def plot_fedfundfutures():
    #ploting - add plots here and in context
    # placeholder, to decide what to add into the drill down
    plot_gdp_index = macro_plot.plot_gdp_index(fake_data)
    context = {'plot_gdp_index': plot_gdp_index} 
    return render_template('fedfundfutures.html', context=context)

