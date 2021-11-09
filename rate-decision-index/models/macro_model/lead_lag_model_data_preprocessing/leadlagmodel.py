import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quandl
from sklearn.preprocessing import MinMaxScaler

quandl.ApiConfig.api_key = "kVFwRskyFgKCs3HURnYV"

class LeadLagModel():
    def __init__(self):
        self.gdp, self.inflation, self.employment, self.others = self.load_data()
        self.full_report = self.get_report()
        
    
    def get_data(self, indicators, start, end):
        df = pd.DataFrame()
        scaler = MinMaxScaler()
        #print("Retrieving data...")
        for indicator in indicators:
            name = "FRED/" + indicator
            mydata = quandl.get(name, start_date=start, end_date=end, collapse='monthly', transform = 'normalize')
            mydata.index = pd.to_datetime(mydata.index).to_period('M')
            mydata.rename(columns={'Value': indicator }, inplace=True)
            mydata[indicator] = ['0.0' if x == '.' else x for x in mydata[indicator]]
            mydata = mydata.resample("M").interpolate()
            mydata[indicator] = pd.to_numeric(mydata[indicator])
            new = scaler.fit_transform(mydata)
            mydata[indicator]=new
            df = pd.concat([df, mydata], axis='columns')
            #print(indicator)
        df = df.fillna(method='ffill')
        #print('Data collected')
        return df
    
    def load_data(self):
        #Excluded outliers
        inflation_indicators = ['MEDCPIM158SFRBCLE', 'T10YIE', 'T5YIE', 'CPILFESL', 'PCEPILFE', 'DFF']
        employment_indicators = ['PAYEMS', 'EMRATIO', 'PRS85006022', 'CES0500000003', 'LNS12032194','LES1252881600Q', 'DFF']       
        gdp_indicators = ['GDP', 'GDPC1', 'A939RX0Q048SBEA', 'DFF']
        others = ['PPIACO','DFF']
        
        start, end ='2004-01-01','2020-01-01'
        
        #gdp data
        print("Retrieving GDP data...")
        gdp = self.get_data(gdp_indicators, start=start, end=end)
        print('GDP Data collected')
        #employment data
        print("Retrieving Employment data...")
        employment = self.get_data(employment_indicators,  start=start, end=end)
        print('Employment Data collected')
        #inflation data
        print("Retrieving Inflation data...")
        inflation = self.get_data(inflation_indicators, start=start, end=end)
        print('Inflation Data collected')
        #others
        print("Retrieving Other data...")
        others = self.get_data(others, start=start, end=end)
        print('Other Data collected')
        
        return gdp, inflation, employment, others
    
    
    def crosscorr(self, datax, datay, lag=0):
        return datax.corr(datay.shift(lag))

    def get_relationship(self, data, lead):
        df = pd.DataFrame()
        for col in data:
            if col == 'DFF':
                continue
            else:
                d1, d2 = data['DFF'], data[col]
                if lead == True:
                    lags = np.arange((0), (6), 1)  # contrained
                elif lead == False:
                    lags = np.arange(-(6), (0), 1)
                rs = np.nan_to_num([self.crosscorr(d1, d2, lag) for lag in lags])
                lst = [col, lags[np.argmax(rs)]]
                a_row=pd.Series([col,lags[np.argmax(rs)] ])
                row_df = pd.DataFrame([a_row])
                df = pd.concat([row_df, df], ignore_index=True)
                print("Calculating relationship between target variable and", col, "...")
        df.rename(columns={0: 'Indicator (Symbol)', 1:'Relationship with Target' }, inplace=True)
        return df
    
    def get_report(self):
        #gdp relationship
        label_gdp = ['(A939RX0Q048SBEA) Real gross domestic product per capita', 
            '(GDPC1) Real gross domestic product', 
            '(GDP) Gross domestic product']
        results_gdp=self.get_relationship(self.gdp, lead=False)
        results_gdp['Indicator name'] = label_gdp
        
        
        #employment relationship 
        label_em = ['Employed full time: Median usual weekly real earnings: Wage and salary workers: 16 years and over',
        'Employment Level - Part-Time for Economic Reasons, All Industries',
        '(CES0500000003) Average Hourly Earnings of All Employees', 
        '(PRS85006022) Nonfarm Business Sector: Average Weekly Hours Worked for All Employed Persons', 
        '(EMRATIO) Employment-Population Ratio',
        '(PAYEMS) All Employees: Total Nonfarm, commonly known as Total Nonfarm Payroll'] 
        results_employment=self.get_relationship(self.employment, lead=False)
        results_employment['Indicator name']=label_em
        
        #inflation relationship
        label_in = ['(PCEPILFE) Personal Consumption Expenditures Excluding Food and Energy',
                   '(CPILFESL) Consumer Price Index for All Urban Consumer',
                   '(T5YIE) 5 Year Breakeven Inflation Rate', 
                   '(T10YIE) 10 Year Breakeven Inflation Rate', 
                   '(MEDCPIM158SFRBCLE) Median Consumer Price Index']
        results_inflation=self.get_relationship(self.inflation, lead=False)
        results_inflation['Indicator name']=label_in
        
        #OTHERS
        label_ot = ['Producer Price Index by Commodity: All Commodities']
        results_others = self.get_relationship(self.others, lead=True)
        results_others['Indicator name']=label_ot
        
        
        big_df = pd.concat([results_gdp, results_employment, results_inflation, results_others])
        big_df.reset_index(inplace=True)
        return big_df
    
        
        
    
        

    
    
        
        
    
        

    

