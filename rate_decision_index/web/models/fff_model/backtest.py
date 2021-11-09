import pandas as pd
import os
from calendar import monthrange
from datetime import datetime,timedelta
import re
import numpy as np
from web.models.fff_model.backtestloader import BacktestLoader
from web.models.fff_model.fff_model import FederalFundsFuture


class Backtest():
    def __init__(self):
        self.loader = BacktestLoader()
    
    def load_month(self, meeting_date:datetime):
        ff_curr = self.loader.get_curr_data(meeting_date)
        ff_prior = self.loader.ff_month_before(meeting_date)
        ff_after = self.loader.ff_month_after(meeting_date)

        prev_month_date = self.cycle_month(meeting_date, step=-1).strftime("%Y-%m")
        fomc_type = 1 if len(self.loader.fomc_dates.loc[prev_month_date]) > 0 else 2
        meeting_date = self.loader.fomc_dates.loc[meeting_date.strftime("%Y-%m")].index[0]
        fff = FederalFundsFuture()
        fff.initiate_model(meeting_date, ff_prior, ff_curr, ff_after, meeting_date, fomc_type)
        
        return fff
    
    def run_month(self, meeting_date:datetime):
        fff = self.load_month(meeting_date)
        no_hike_prob, hike_prob = fff.calculate_hike_prob()
        
        prob_change = [no_hike_prob, hike_prob]
                        
        return prob_change,fff

    def find_range(self,implied_rate, probs):
        int_ranges = [0.25,0.5,0.75,1,1.25,1.5,1.75,2]
        values = [0,0,0,0,0,0,0,0]
        for i in range(len(int_ranges)-1):
            if int_ranges[i] > implied_rate:
                level = i
                break

        for prob in range(len(probs)):
            values[level + prob] = probs[prob]
        return values
    
    def predict(self):
        today = datetime.now().strftime("%Y-%m")
        meeting_dates = self.loader.fomc_dates.loc[today:]
        all_predictions = {}
        for dt in meeting_dates.index:
            print(f"Loading: {dt} FOMC Meeting...")
            dt = pd.to_datetime(dt)
            probs,fff = self.run_month(dt)
            result = self.carry(probs)

            implied_rate = fff.implied_rate
            v = self.find_range(implied_rate, result)


            all_predictions[dt] = v
        
        final_result = pd.DataFrame.from_dict(all_predictions).T
        final_result.columns = ['0-25 BPS','25-50 BPS','50-75 BPS',
                                '75-100 BPS','100-125 BPS','125-150 BPS',
                                '150-175 BPS', '175-200 BPS']
        return final_result
    
    def carry(self, sample,cap=1):
        result = sample.copy()
        
        if result[0] < 0:
            result.append(0)

        result = np.array([result])
        for c in range(1,result.shape[1]):
            result[:,c] += np.maximum(result[:,c-1]-cap,0)
        result[:,:-1]  = np.minimum(result[:,:-1],cap)
        
        if result[:,0] < 0:
            result[:,0] = 0 
            for i in range(1,result.shape[1]-1):
                if result[:,i] == 1:
                    result[:,i] = 1 - result[:,i+1]
        return result[0].tolist()
    
    def cycle_month(self,date: datetime, step):
        new_date = date + step * timedelta(days=monthrange(date.year, date.month)[1] )
        return new_date

     