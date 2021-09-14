import pandas as pd
from fred import Fred
fred_api = '18fb1a5955cab2aae08b90a2ff0f6e42'
fred = Fred(api_key=fred_api, response_type = 'df')

def fetch_data(type:str):
    pass
    # TODO: handle our data i/o, fetch data using API or what