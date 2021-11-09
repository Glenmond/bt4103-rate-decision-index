import os
import json
import time
import requests
from datetime import datetime
import dateutil
import pandas as pd
import pickle

apikey = os.getenv('NYTIMES_APIKEY', 'w3R4zcqQEA87fNdWNj866OzJ03SvgOwu')


class News():

    def __init__(self):
        # Set arguments to internal variables
        self.content_type = "news"
        self.base_dir = '../data/sentiment_data/extract/' # '../data/db/pickle/extract/'
        self.df = None
        self.start = None
        self.end = None


    def send_request(self, date):
        '''Sends a request to the NYT Archive API for given date.'''
        base_url = 'https://api.nytimes.com/svc/archive/v1/'
        url = base_url + '/' + date[0] + '/' + date[1] + '.json?api-key=' + apikey
        response = requests.get(url).json()
        time.sleep(6)
        return response


    def is_valid(self, article, date):
        '''An article is only worth checking if it is in range, and has a snippet.
        Snippet > Headlines'''
        is_in_range = date > self.start and date < self.end
        has_snippet = type(article) == dict and 'snippet' in article.keys()
        return is_in_range and has_snippet


    def parse_response(self, response):
        '''Parses and returns response as pandas data frame.'''
        data = {'article snippet': [],  
            'date': [], 
            'doc_type': [],
            'material_type': [],
            'section': [],
            'keywords': []}
        
        articles = response['response']['docs'] 
        for article in articles: # For each article, make sure it falls within our date range
            date = dateutil.parser.parse(article['pub_date']).date()
            if self.is_valid(article, date):
                data['date'].append(date)
                data['article snippet'].append(article['snippet']) 
                if 'section' in article:
                    data['section'].append(article['section_name'])
                else:
                    data['section'].append(None)
                data['doc_type'].append(article['document_type'])
                if 'type_of_material' in article: 
                    data['material_type'].append(article['type_of_material'])
                else:
                    data['material_type'].append(None)
                keywords = [keyword['value'] for keyword in article['keywords'] if keyword['name'] == 'subject']
                data['keywords'].append(keywords)
        return pd.DataFrame(data) 

    def pickle_dump_df(self, filename="output.pickle"):
        '''
        Dump df to a pickle file
        '''
        filepath = self.base_dir + filename
        print("")
        print("Writing to ", filepath)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as output_file:
            pickle.dump(self.df, output_file)
    
    def get_contents(self, from_year):
        '''Sends and parses request/response to/from NYT Archive API for given dates.'''
        
        # Set arguments to internal variables
        self.end = datetime.date(datetime.today())
        self.start = datetime.date(datetime(from_year, 1, 1))

        dates = [x.split(' ') for x in pd.date_range(self.start, self.end, freq='MS').strftime("%Y %-m").tolist()]
        ldf = []
        total = 0
        print("Getting articles for news..." + 'Date range: ' + str(dates[0]) + ' to ' + str(dates[-1]))

        for date in dates:
            response = self.send_request(date)
            df = self.parse_response(response)
            total += len(df)
            ldf.append(df)
            print('Saving article/' + date[0] + '-' + date[1] + ' to pickle')
        print(f'There are total {total} number of articles collected')
        
        self.df = pd.concat(ldf, ignore_index=True, sort=True)
        return self.df