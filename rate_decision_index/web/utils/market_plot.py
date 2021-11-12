import pandas as pd
import datetime
import plotly
import plotly.graph_objects as go
import plotly.express as px
import json
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime

## Time series of market sentiments (drill down)
def plot_top_n_trigram(data):
    corpus = data
    n=3
    vec = CountVectorizer(ngram_range=(n, n)).fit([corpus])
    bag_of_words = vec.transform([corpus])
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    df_plot = pd.DataFrame(words_freq[:10], columns = ['Word/Phrase' , 'Count'])
    fig = px.bar(df_plot, x='Word/Phrase', y='Count')
    
    fig.update_layout(
        font_family="Courier New",
        font_color="black",
        title_font_family="Times New Roman Bold",
        title_font_color="black",
        title_text='Top 10 Most Common Words/Phrases', 
        title_x=0.5
    )
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json

#helper function
def concat_list(series):
    """Concat list of strings into one string"""
    
    all_words = ''
    for s in series['lemmatizedSentences'].iteritems():
        words = ' '.join(s[1])
        all_words = all_words + words

    return all_words

def get_top_n_gram_mins(data, date):
    date_i = int(date)
    date_next = date_i+ 1
    date_next = str(date_next)
    date_now = datetime.strptime(str(date), '%Y')
    date_next = datetime.strptime(date_next, '%Y')
    print(data.head())
    
    mins_df = data[(data.date >= date_now) & (data.date <= date_next)]
    ## Data
    #n=1
    vec = CountVectorizer(ngram_range=(1, 1)).fit([concat_list(mins_df)])
    bag_of_words = vec.transform([concat_list(mins_df)])
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    df_plot_1 = pd.DataFrame(words_freq[:5], columns = ['Words' , 'Count'])

    #n=2
    vec = CountVectorizer(ngram_range=(2, 2)).fit([concat_list(mins_df)])
    bag_of_words = vec.transform([concat_list(mins_df)])
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    df_plot_2 = pd.DataFrame(words_freq[:5], columns = ['Words' , 'Count'])

    #n=3
    vec = CountVectorizer(ngram_range=(3, 3)).fit([concat_list(mins_df)])
    bag_of_words = vec.transform([concat_list(mins_df)])
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    df_plot_3 = pd.DataFrame(words_freq[:5], columns = ['Words' , 'Count'])

    #n=4
    vec = CountVectorizer(ngram_range=(4, 4)).fit([concat_list(mins_df)])
    bag_of_words = vec.transform([concat_list(mins_df)])
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    df_plot_4 = pd.DataFrame(words_freq[:5], columns = ['Words' , 'Count'])

    #n=5
    vec = CountVectorizer(ngram_range=(5, 5)).fit([concat_list(mins_df)])
    bag_of_words = vec.transform([concat_list(mins_df)])
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    df_plot_5 = pd.DataFrame(words_freq[:5], columns = ['Words' , 'Count'])

    ## Plot
    plot = go.Figure(data=[go.Bar(
        name='Top 5 Most Common Uni-gram in FOMC Minutes',
        x=df_plot_1.Words.tolist(),
        y=df_plot_1.Count.tolist(),
        marker_color='#401664' #change color of bars
    ),
        go.Bar(
        name='Top 5 Most Common Bi-gram in FOMC Minutes',
        x=df_plot_2.Words.tolist(),
        y=df_plot_2.Count.tolist(),
        marker_color='#D71C2B' #change color of bars
    ),
        go.Bar(
        name='Top 5 Most Common Tri-gram in FOMC Minutes',
        x=df_plot_3.Words.tolist(),
        y=df_plot_3.Count.tolist(),
        marker_color='#EE2033' #change color of bars
    ),
        go.Bar(
        name='Top 5 Most Common 4-gram in FOMC Minutes',
        x=df_plot_4.Words.tolist(),
        y=df_plot_4.Count.tolist(),
        marker_color='#F78E99' #change color of bars
    ),
        go.Bar(
        name='Top 5 Most Common 5-gram in FOMC Minutes',
        x=df_plot_5.Words.tolist(),
        y=df_plot_5.Count.tolist(),
        marker_color='#FBC9CF' #change color of bars
    )
    ])
    
    # Drop Down Menu
    plot.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=list([
                    dict(label="All",
                         method="update",
                         args=[{"visible": [True, True, True, True, True]},
                               {"title": "Most Common Words and Phrases in FOMC Minutes"}]),
                    dict(label="1",
                         method="update",
                         args=[{"visible": [True, False, False, False, False]},
                               {"title": "Top 5 Most Common Uni-gram in FOMC Minutes",
                                }]),
                    dict(label="2",
                         method="update",
                         args=[{"visible": [False, True, False, False, False]},
                               {"title": "Top 5 Most Common Bi-gram in FOMC Minutes",
                                }]),
                    dict(label="3",
                         method="update",
                         args=[{"visible": [False, False, True, False, False]},
                               {"title": "Top 5 Most Common Tri-gram in FOMC Minutes",
                                }]),
                    dict(label="4",
                         method="update",
                         args=[{"visible": [False, False, False, True, False]},
                               {"title": "Top 5 Most Common 4-gram in FOMC Minutes",
                                }]),
                    dict(label="5",
                         method="update",
                         args=[{"visible": [False, False, False, False, True]},
                               {"title": "Top 5 Most Common 5-gram in FOMC Minutes",
                                }]),
                ]),
            )
        ])
    
    # Aesthetic 
    plot.update_layout(
        font_family="Courier New",
        font_color="black",
        title_font_family="Times New Roman Bold",
        title_font_color="black",
        title_text='Most Common Words and Phrases in FOMC Minutes for ' + str(date), 
        title_x=0.5,
        plot_bgcolor = 'white'
    )
    plot.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#ECECEC', zeroline=True, zerolinecolor='lightgrey')
    plot.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#ECECEC', zeroline=True, zerolinecolor='lightgrey')
    
    plot_json = json.dumps(plot, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json

def get_top_n_gram_st(data, date):
    date_i = int(date)
    date_next = date_i+ 1
    date_next = str(date_next)
    date_now = datetime.strptime(str(date), '%Y')
    date_next = datetime.strptime(date_next, '%Y')
    
    st_df = data[(data.date >= date_now) & (data.date <= date_next)]
    ## Data
    #n=1
    vec = CountVectorizer(ngram_range=(1, 1)).fit([concat_list(st_df)])
    bag_of_words = vec.transform([concat_list(st_df)])
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    df_plot_1 = pd.DataFrame(words_freq[:5], columns = ['Words' , 'Count'])

    #n=2
    vec = CountVectorizer(ngram_range=(2, 2)).fit([concat_list(st_df)])
    bag_of_words = vec.transform([concat_list(st_df)])
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    df_plot_2 = pd.DataFrame(words_freq[:5], columns = ['Words' , 'Count'])

    #n=3
    vec = CountVectorizer(ngram_range=(3, 3)).fit([concat_list(st_df)])
    bag_of_words = vec.transform([concat_list(st_df)])
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    df_plot_3 = pd.DataFrame(words_freq[:5], columns = ['Words' , 'Count'])

    #n=4
    vec = CountVectorizer(ngram_range=(4, 4)).fit([concat_list(st_df)])
    bag_of_words = vec.transform([concat_list(st_df)])
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    df_plot_4 = pd.DataFrame(words_freq[:5], columns = ['Words' , 'Count'])

    #n=5
    vec = CountVectorizer(ngram_range=(5, 5)).fit([concat_list(st_df)])
    bag_of_words = vec.transform([concat_list(st_df)])
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    df_plot_5 = pd.DataFrame(words_freq[:5], columns = ['Words' , 'Count'])

    ## Plot
    plot = go.Figure(data=[go.Bar(
        name='Top 5 Most Common Uni-gram in FOMC Statements',
        x=df_plot_1.Words.tolist(),
        y=df_plot_1.Count.tolist(),
        marker_color='#401664' #change color of bars
    ),
        go.Bar(
        name='Top 5 Most Common Bi-gram in FOMC Statements',
        x=df_plot_2.Words.tolist(),
        y=df_plot_2.Count.tolist(),
        marker_color='#D71C2B' #change color of bars
    ),
        go.Bar(
        name='Top 5 Most Common Tri-gram in FOMC Statements',
        x=df_plot_3.Words.tolist(),
        y=df_plot_3.Count.tolist(),
        marker_color='#EE2033' #change color of bars
    ),
        go.Bar(
        name='Top 5 Most Common 4-gram in FOMC Statements',
        x=df_plot_4.Words.tolist(),
        y=df_plot_4.Count.tolist(),
        marker_color='#F78E99' #change color of bars
    ),
        go.Bar(
        name='Top 5 Most Common 5-gram in FOMC Statements',
        x=df_plot_5.Words.tolist(),
        y=df_plot_5.Count.tolist(),
        marker_color='#FBC9CF' #change color of bars
    )
    ])
    
    # Drop Down Menu
    plot.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=list([
                    dict(label="All",
                         method="update",
                         args=[{"visible": [True, True, True, True, True]},
                               {"title": "Most Common Words and Phrases in FOMC Statements"}]),
                    dict(label="1",
                         method="update",
                         args=[{"visible": [True, False, False, False, False]},
                               {"title": "Top 5 Most Common Uni-gram in FOMC Statements",
                                }]),
                    dict(label="2",
                         method="update",
                         args=[{"visible": [False, True, False, False, False]},
                               {"title": "Top 5 Most Common Bi-gram in FOMC Statements",
                                }]),
                    dict(label="3",
                         method="update",
                         args=[{"visible": [False, False, True, False, False]},
                               {"title": "Top 5 Most Common Tri-gram in FOMC Statements",
                                }]),
                    dict(label="4",
                         method="update",
                         args=[{"visible": [False, False, False, True, False]},
                               {"title": "Top 5 Most Common 4-gram in FOMC Statements",
                                }]),
                    dict(label="5",
                         method="update",
                         args=[{"visible": [False, False, False, False, True]},
                               {"title": "Top 5 Most Common 5-gram in FOMC Statements",
                                }]),
                ]),
            )
        ])
    
    # Aesthetic 
    plot.update_layout(
        font_family="Courier New",
        font_color="black",
        title_font_family="Times New Roman Bold",
        title_font_color="black",
        title_text='Most Common Words and Phrases in FOMC Statements for ' + str(date), 
        title_x=0.5,
        plot_bgcolor = 'white'
    )
    
    plot.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#ECECEC', zeroline=True, zerolinecolor='lightgrey')
    plot.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#ECECEC', zeroline=True, zerolinecolor='lightgrey')
    plot_json = json.dumps(plot, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json

def get_top_n_gram_news(data, date):
    date_i = int(date)
    date_next = date_i+ 1
    date_next = str(date_next)
    date_now = datetime.strptime(str(date), '%Y')
    date_next = datetime.strptime(date_next, '%Y')
    
    news_df = data[(data.date >= date_now) & (data.date <= date_next)]
    ## Data
    #n=1
    vec = CountVectorizer(ngram_range=(1, 1)).fit([concat_list(news_df)])
    bag_of_words = vec.transform([concat_list(news_df)])
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    df_plot_1 = pd.DataFrame(words_freq[:5], columns = ['Words' , 'Count'])

    #n=2
    vec = CountVectorizer(ngram_range=(2, 2)).fit([concat_list(news_df)])
    bag_of_words = vec.transform([concat_list(news_df)])
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    df_plot_2 = pd.DataFrame(words_freq[:5], columns = ['Words' , 'Count'])

    #n=3
    vec = CountVectorizer(ngram_range=(3, 3)).fit([concat_list(news_df)])
    bag_of_words = vec.transform([concat_list(news_df)])
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    df_plot_3 = pd.DataFrame(words_freq[:5], columns = ['Words' , 'Count'])

    #n=4
    vec = CountVectorizer(ngram_range=(4, 4)).fit([concat_list(news_df)])
    bag_of_words = vec.transform([concat_list(news_df)])
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    df_plot_4 = pd.DataFrame(words_freq[:5], columns = ['Words' , 'Count'])

    #n=5
    vec = CountVectorizer(ngram_range=(5, 5)).fit([concat_list(news_df)])
    bag_of_words = vec.transform([concat_list(news_df)])
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    df_plot_5 = pd.DataFrame(words_freq[:5], columns = ['Words' , 'Count'])

    ## Plot
    plot = go.Figure(data=[go.Bar(
        name='Top 5 Most Common Uni-gram in the News',
        x=df_plot_1.Words.tolist(),
        y=df_plot_1.Count.tolist(),
        marker_color='#401664' #change color of bars
    ),
        go.Bar(
        name='Top 5 Most Common Bi-gram in the News',
        x=df_plot_2.Words.tolist(),
        y=df_plot_2.Count.tolist(),
        marker_color='#D71C2B' #change color of bars
    ),
        go.Bar(
        name='Top 5 Most Common Tri-gram in the News',
        x=df_plot_3.Words.tolist(),
        y=df_plot_3.Count.tolist(),
        marker_color='#EE2033' #change color of bars
    ),
        go.Bar(
        name='Top 5 Most Common 4-gram in the News',
        x=df_plot_4.Words.tolist(),
        y=df_plot_4.Count.tolist(),
        marker_color='#F78E99' #change color of bars
    ),
        go.Bar(
        name='Top 5 Most Common 5-gram in the News',
        x=df_plot_5.Words.tolist(),
        y=df_plot_5.Count.tolist(),
        marker_color='#FBC9CF' #change color of bars
    )
    ])
    
    # Drop Down Menu
    plot.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=list([
                    dict(label="All",
                         method="update",
                         args=[{"visible": [True, True, True, True, True]},
                               {"title": "Most Common Words and Phrases in the News"}]),
                    dict(label="1",
                         method="update",
                         args=[{"visible": [True, False, False, False, False]},
                               {"title": "Top 5 Most Common Uni-gram in the News",
                                }]),
                    dict(label="2",
                         method="update",
                         args=[{"visible": [False, True, False, False, False]},
                               {"title": "Top 5 Most Common Bi-gram in the News",
                                }]),
                    dict(label="3",
                         method="update",
                         args=[{"visible": [False, False, True, False, False]},
                               {"title": "Top 5 Most Common Tri-gram in the News",
                                }]),
                    dict(label="4",
                         method="update",
                         args=[{"visible": [False, False, False, True, False]},
                               {"title": "Top 5 Most Common 4-gram in the News",
                                }]),
                    dict(label="5",
                         method="update",
                         args=[{"visible": [False, False, False, False, True]},
                               {"title": "Top 5 Most Common 5-gram in the News",
                                }]),
                ]),
            )
        ])
    
    # Aesthetic 
    plot.update_layout(
        font_family="Courier New",
        font_color="black",
        title_font_family="Times New Roman Bold",
        title_font_color="black",
        title_text='Most Common Words and Phrases in the News for ' + str(date), 
        title_x=0.5,
        plot_bgcolor = 'white'
    )
    
    plot.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#ECECEC', zeroline=True, zerolinecolor='lightgrey')
    plot.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#ECECEC', zeroline=True, zerolinecolor='lightgrey')
    plot_json = json.dumps(plot, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json

def plot_hd_ts(data):
    df_senti = data

    plot = go.Figure(data=[go.Scatter(
        name='FOMC Statements',
        x=df_senti.Date.tolist(),
        y=df_senti.Score_Statement.tolist(),
        marker_color='#401664' #change color of line
    ),
        go.Scatter(
        name='FOMC Minutes',
        x=df_senti.Date.tolist(),
        y=df_senti.Score_Minutes.tolist(),
        marker_color='#19AADE' #change color of line
    ),
        go.Scatter(
        name='News',
        x=df_senti.Date.tolist(),
        y=df_senti.Score_News.tolist(),
        marker_color='#F78E99' #change color of line
    )                       

    ])

    plot.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=list([
                    dict(label="Combined",
                         method="update",
                         args=[{"visible": [True, True, True]},
                               {"title": "Time Series of Sentiment Scores"}]),
                    dict(label="FOMC Statements",
                         method="update",
                         args=[{"visible": [True, False, False]},
                               {"title": "Time Series of FOMC Statements Sentiments Scores",
                                }]),
                    dict(label="FOMC Minutes",
                         method="update",
                         args=[{"visible": [False, True, False]},
                               {"title": "Time Series of FOMC Minutes Sentiments Scores",
                                }]),
                    dict(label="News",
                         method="update",
                         args=[{"visible": [False, False, True]},
                               {"title": "Time Series of News Sentiments Scores",
                                }]),
                ]),
            )
        ])

    plot.update_layout(
            font_family="Courier New",
            font_color="black",
            title_font_family="Times New Roman Bold",
            title_font_color="black",
            title_text='Time Series of Sentiment Scores', 
            title_x=0.5, plot_bgcolor = 'white'
        )
    plot.update_xaxes(rangeslider_visible=True, showgrid=True, gridwidth=1, gridcolor='#ECECEC', zeroline=True, zerolinecolor='lightgrey')
    plot.update_yaxes(range=[-1.1, 1.1], showgrid=True, gridwidth=1, gridcolor='#ECECEC', zeroline=True, zerolinecolor='lightgrey')

    plot_json = json.dumps(plot, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json