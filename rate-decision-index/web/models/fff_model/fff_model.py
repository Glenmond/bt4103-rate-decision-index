import pandas as pd
import os
from calendar import monthrange, month_name
from datetime import datetime,timedelta
import re
import numpy as np


class FederalFundsFuture():
        
    def initiate_model(self, 
                       date: datetime, 
                       ff_prior, 
                       ff_curr, 
                       ff_after, 
                       meeting_date,
                       fomc_type):
        self.date = date
        self.ff_prior = ff_prior #ff.monthbefore
        self.ff_curr = ff_curr #ff.meeting_month
        self.ff_after = ff_after
        self.days =  monthrange(meeting_date.year, meeting_date.month)[1] 
        self.meeting_date = meeting_date
        self.type = fomc_type
        
        self.meeting_day = meeting_date.day 
        self.ff_curr = ff_curr
        self.implied_rate = 100 - ff_curr
        
        self.calculate_ffer_start_end()
        
    def set_probabilities(self, decrease, unchanged, increase):
        self.ffer_decrease_prob = decrease
        self.ffer_unchanged_prob  = unchanged
        self.ffer_increase_prob = increase
        
    def calculate_ffer_start_end(self):
        if self.type == 2:
            self.ffer_start = self.calculate_ffer_start()
            self.ffer_end = self.calculate_ffer_end()
        else:
            self.ffer_end = self.calculate_ffer_end()
            self.ffer_start = self.calculate_ffer_start()
            
    def calculate_hike_prob(self, increment=0.25):
        hike_prob = (self.ffer_end-self.ffer_start)/increment
        no_hike_prob = 1 - hike_prob
        return no_hike_prob, hike_prob
    
    
    def calculate_rate_level_prob(self, prev_increase, prev_decrease):
        hike_prob, no_hike_prob = self.calculate_hike_prob()
        
        ffer_decrease_prob = prev_decrease * (1-hike_prob)
        ffer_unchanged_prob = prev_increase * (1-hike_prob) + prev_decrease * (hike_prob)
        ffer_increase_prob = prev_increase * hike_prob
        
        self.set_probabilities(ffer_decrease_prob, ffer_unchanged_prob, ffer_increase_prob)
        
    def calculate_ffer_start(self):
        if self.type == 2:
            return 100-self.ff_prior
        else:
            n = self.days
            m = self.meeting_day
            ir = self.implied_rate
            #print(f"({n}/{m}) * ({ir} - ({self.ffer_end}*(({n}-{m})/{n})))")
            return (n/m) * (ir - (self.ffer_end*((n-m)/n)))
    
    def calculate_ffer_end(self):
        if self.type == 2:
            n = self.days
            m = self.meeting_day
            ir = self.implied_rate
            #print(f"{n}/({n}-{m})) * ({ir} - ({m}/{n})*{self.ffer_start})")
            return (n/(n-m)) * (ir - (m/n)*self.ffer_start)
        else:
            return 100- self.ff_after
        
    def __str__(self):
        str_definition = f"""
        FEDERAL FUNDS MODEL for {self.date.strftime("%d-%m-%Y")}
        TYPE: {self.type}
        
        FF Month Before: {round(self.ff_prior,2)}
        FF Current Month: {round(self.ff_curr,2)}
        FF Implied Rate: {round(self.implied_rate,2)}
        
        FFER End: {round(self.ffer_end,2)}
        FFER Start: {round(self.ffer_start,2)}
        
        """
        return str_definition
        