import pandas as pd
from fredapi import Fred
fred_api = '18fb1a5955cab2aae08b90a2ff0f6e42'
fred = Fred(api_key=fred_api)

def fetch_data():
    # Fetches data from FRED API, stores it in a dataframe
    commodities_params = { 
        'observation_start':'2003-01-02',
        'observation_end':'2021-04-01',
        'units':'lin',
        'frequency':'m',
        'aggregation_method': 'eop',   
    }
    commodities = ('PPIACO', commodities_params)

    # one period lag 
    # one period lag means obsevation starts and ends one period later
    # nonfarm_params={ 
    #     'observation_start':'2003-01-02',
    #     'observation_end':'2021-05-01',
    #     'units':'lin',
    #     'frequency':'m',
    #     'aggregation_method': 'eop',   
    # }
    # nonfarm=('PAYEMS',nonfarm_params)

    # breakeven_params = { 
    #     'observation_start':'2003-01-02',
    #     'observation_end':'2021-05-01',
    #     'units':'lin',
    #     'frequency':'m',
    #     'aggregation_method': 'eop',   
    # }
    # breakeven = ('T5YIE', breakeven_params)

    real_gdp_params = { 
        'observation_start':'2003-01-02',
        'observation_end':'2021-05-01',
        'units':'lin',
        'frequency':'q',
        'aggregation_method': 'eop',   
    }
    real_gdp = ('GDPC1', real_gdp_params)

    # two period lag
    # two period lag means obsevation starts and ends two periods later since we have to shift forward
    median_cpi_params ={ 
        'observation_start':'2003-01-02',
        'observation_end':'2021-06-01',
        'units':'lin',
        'frequency':'m',
        'aggregation_method': 'eop',   
    }
    median_cpi = ('MEDCPIM158SFRBCLE', median_cpi_params)

    # three period lag
    # three period lag means obsevation starts and ends three periods later since we have to shift forward
    em_ratio_params = { 
        'observation_start':'2003-01-02',
        'observation_end':'2021-07-01',
        'units':'lin',
        'frequency':'m',
        'aggregation_method': 'eop',   
    }
    em_ratio = ('EMRATIO', em_ratio_params)

    # five period lag
    med_wages_params = { 
        'observation_start':'2003-01-02',
        'observation_end':'2021-09-01',
        'units':'lin',
        'frequency':'q',
        'aggregation_method': 'eop',   
    }
    med_wages = ('LES1252881600Q', med_wages_params)


    # two period lead
    # two period lead means observation starts and ends two periods earlier since we have to shift backward
    # neworder_params = { 
    #     'observation_start':'2002-11-02', # 2 period lead so observation starts two periods earlier
    #     'observation_end':'2021-04-01', # 2 period lead so observation ends two periods earlier
    #     'units':'lin',
    #     'frequency':'m',
    #     'aggregation_method': 'eop',   
    # }
    # new_order=('NEWORDER', neworder_params)

    # 4 period lead
    # four period lead means observation starts and ends four periods earlier since we have to shift backward
    # job_opening_manu_params={
    #     'observation_start':'2002-09-02', 
    #     'observation_end':'2021-04-01',
    #     'units':'lin',
    #     'frequency':'m',
    #     'aggregation_method': 'eop',
    # }
    # job_opening_manu = ('JTS3000JOL', job_opening_manu_params)

    # 5 period lead
    # five period lead means observation starts and ends five periods earlier since we have to shift backward
    # permit_params = {
    #     'observation_start':'2002-08-02', 
    #     'observation_end':'2021-04-01',
    #     'units':'lin',
    #     'frequency':'m',
    #     'aggregation_method': 'eop',
    # }
    # permit = ('PERMIT', permit_params)

    # amtmno_params={
    #     'observation_start':'2002-08-02', 
    #     'observation_end':'2021-04-01',
    #     'units':'lin',
    #     'frequency':'m',
    #     'aggregation_method': 'eop',
    # }
    # amtmno = ('AMTMNO', amtmno_params)

    # dgorder_params = {
    #     'observation_start':'2002-08-02', 
    #     'observation_end':'2021-04-01',
    #     'units':'lin',
    #     'frequency':'m',
    #     'aggregation_method': 'eop',
    # }
    # dgorder=('DGORDER', dgorder_params)

    maturity_minus_three_month_params = {
        'observation_start':'2002-08-02', 
        'observation_end':'2021-04-01',
        'units':'lin',
        'frequency':'m',
        'aggregation_method': 'eop',
    }
    maturity_minus_three_month = ('T10Y3M', maturity_minus_three_month_params)

    # normal_indicators = [nonfarm, breakeven, real_gdp, median_cpi, em_ratio, new_order,
    #                     job_opening_manu, permit, amtmno, dgorder, maturity_minus_three_month, commodities, med_wages]

    indicators = [commodities, real_gdp, median_cpi, em_ratio, med_wages, maturity_minus_three_month]

    df = pd.DataFrame()

    # target value
    fed_fund_rate = fred.get_series('DFF',**{'observation_start':'2003-01-02', # T5YIE only starts at 2003-01-02, and is shifted by 1 lag forward, so our observation starts from 2002-12-02
                                             'observation_end':'2021-04-01', # GDPC1 is only collected until 2020-04-01
                                             'frequency':'m',
                                             'aggregation_method': 'eop'})
    
    fed_fund_rate.index = pd.to_datetime(fed_fund_rate.index).to_period('M')
    df['target'] = fed_fund_rate.to_numpy()
    df.index = fed_fund_rate.index
    for series_id, params in indicators:
        # Get the data from FRED, convert to pandas DataFrame
        indicator = fred.get_series(series_id, **params)
        indicator = indicator.to_frame().set_axis([series_id],axis='columns')
        # fill in data with '0.0' that is presented as just '.'
        indicator[series_id] = ['0.0' if x == '.' else x for x in indicator[series_id]]
        # turn the value into numeric
        indicator[series_id] = pd.to_numeric(indicator[series_id])
        indicator.index = pd.to_datetime(indicator.index).to_period('M')
        indicator = indicator.resample("M").interpolate()
        
        if series_id in ('PAYEMS', 'T5YIE', 'GDPC1'): # align 1 lag
            indicator = indicator.shift(-1)[:-1]
            
        if series_id in ('MEDCPIM158SFRBCLE'): # align 2 lag
            indicator = indicator.shift(-2)[:-2]
            indicator.rename(columns={'MEDCPIM158SFRBCLE':'MEDCPI'}, inplace=True)
            
        if series_id in ('EMRATIO'): # align 3 lag
            indicator = indicator.shift(-3)[:-3]
            
        if series_id in ('LES1252881600Q'): # align 5 lag
            indicator = indicator.shift(-5)[:-5]
            indicator.rename(columns={'LES1252881600Q':'MEDWAGES'}, inplace=True)
       
        # if series_id in ('NEWORDER'): # align 2 lead
        #     indicator = indicator.shift(2)[2:]
            
        # if series_id in ('JTS3000JOL'): # align 4 lead
        #     indicator = indicator.shift(4)[4:]
        
        if series_id in ('PERMIT', 'AMTMNO', 'DGORDER', 'T10Y3M'): #align 5 lead
            indicator = indicator.shift(5)[5:]
            #print(indicator)
        
        # join the dataframes together
        df = pd.concat([indicator,df], axis='columns')

    # DO DATA IMPUTATION FOR POSSIBLE NAN VALUES
    df = df.fillna(method='ffill')
    return df
    
    # remove outliers
    z_scores = zscore(df)
    abs_z_scores = np.abs(z_scores)
    threshold = 2.5
    filtered_entries = (abs_z_scores < threshold).all(axis=1)
    df = df[filtered_entries]

    return df