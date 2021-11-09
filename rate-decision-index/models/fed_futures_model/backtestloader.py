import pandas as pd
import os
from datetime import datetime,timedelta
import yfinance as yf

from models.extract.import_data import download_fed_futures_historical, download_fomc_dates

class BacktestLoader():
    def __init__(self):
        self.futures = self.load_futures_historical()
        self.fomc_dates = self.get_fomc_dates()
        self.models = {}
        
        self.contract_codes = {
            "January" :"F",
            "February" :"G",
            "March" :"H",
            "April" :"J",
            "May" :"K",
            "June" :"M",
            "July" :"N",
            "August" :"Q",
            "September" :"U",
            "October" :"V",
            "November" :"X",
            "December" :"Z"
        }
        
    def load_futures_historical(self):
        # Futures: full historical + forward data
        # download from barchart
        futures = download_fed_futures_historical()        
        return futures
    
    def load_meeting_futures_data(self, meeting_date,period='max'):
        year = meeting_date.strftime('%y')
        code = self.contract_codes[meeting_date.strftime("%B")]
        ticker = f"ZQ{code}{year}.CBT"
        df = yf.Ticker(ticker).history(period=period)

        return df
    
    def get_fomc_dates(self):
        df = download_fomc_dates()
        return df
            
    def get_targets_data(self, date:datetime):
        upper_target = self.upper_target.loc[date]
        lower_target = self.lower_target.loc[date]
        return upper_target, lower_target
    
    def get_curr_data(self, date:datetime):
        return self.load_meeting_futures_data(date, "1mo")['Close'][-1]
    
    def ff_month_after(self, date:datetime):
        month_after = date + timedelta(days=30)
        return self.load_meeting_futures_data(month_after, "1mo")['Close'][-1]
    
    def ff_month_before(self, date: datetime):
        month_before = date - timedelta(days=30)
        return self.futures.loc[month_before.strftime("%Y-%m")]['Last'][0]
            
    