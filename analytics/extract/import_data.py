import sys
import datetime
import pandas as pd
import numpy as np

from scipy.stats import zscore

from bs4 import BeautifulSoup
import requests
import re
from fredapi import Fred

try:
    from fomc_data.FomcStatement import FomcStatement
    from fomc_data.FomcMinutes import FomcMinutes
    from news_data.News import News
except ModuleNotFoundError: 
    from .fomc_data.FomcStatement import FomcStatement
    from .fomc_data.FomcMinutes import FomcMinutes
    from .news_data.News import News


batch_id = datetime.date.today().strftime("%y%m%d")

fred_api = "18fb1a5955cab2aae08b90a2ff0f6e42"
fred = Fred(api_key=fred_api)

def fetch_data():
    # Fetches data from FRED API, stores it in a dataframe
    commodities_params = {
        "observation_start": "2003-01-02",
        "observation_end": "2021-04-01",
        "units": "lin",
        "frequency": "m",
        "aggregation_method": "eop",
    }
    commodities = ("PPIACO", commodities_params)

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
        "observation_start": "2003-01-02",
        "observation_end": "2021-05-01",
        "units": "lin",
        "frequency": "q",
        "aggregation_method": "eop",
    }
    real_gdp = ("GDPC1", real_gdp_params)

    # two period lag
    # two period lag means obsevation starts and ends two periods later since we have to shift forward
    median_cpi_params = {
        "observation_start": "2003-01-02",
        "observation_end": "2021-06-01",
        "units": "lin",
        "frequency": "m",
        "aggregation_method": "eop",
    }
    median_cpi = ("MEDCPIM158SFRBCLE", median_cpi_params)

    # three period lag
    # three period lag means obsevation starts and ends three periods later since we have to shift forward
    em_ratio_params = {
        "observation_start": "2003-01-02",
        "observation_end": "2021-07-01",
        "units": "lin",
        "frequency": "m",
        "aggregation_method": "eop",
    }
    em_ratio = ("EMRATIO", em_ratio_params)

    # five period lag
    med_wages_params = {
        "observation_start": "2003-01-02",
        "observation_end": "2021-09-01",
        "units": "lin",
        "frequency": "q",
        "aggregation_method": "eop",
    }
    med_wages = ("LES1252881600Q", med_wages_params)

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
        "observation_start": "2002-08-02",
        "observation_end": "2021-04-01",
        "units": "lin",
        "frequency": "m",
        "aggregation_method": "eop",
    }
    maturity_minus_three_month = ("T10Y3M", maturity_minus_three_month_params)

    # normal_indicators = [nonfarm, breakeven, real_gdp, median_cpi, em_ratio, new_order,
    #                     job_opening_manu, permit, amtmno, dgorder, maturity_minus_three_month, commodities, med_wages]

    indicators = [
        commodities,
        real_gdp,
        median_cpi,
        em_ratio,
        med_wages,
        maturity_minus_three_month,
    ]

    df = pd.DataFrame()

    # target value
    fed_fund_rate = fred.get_series(
        "DFF",
        **{
            "observation_start": "2003-01-02",  # T5YIE only starts at 2003-01-02, and is shifted by 1 lag forward, so our observation starts from 2002-12-02
            "observation_end": "2021-04-01",  # GDPC1 is only collected until 2020-04-01
            "frequency": "m",
            "aggregation_method": "eop",
        },
    )

    fed_fund_rate.index = pd.to_datetime(fed_fund_rate.index).to_period("M")
    df["target"] = fed_fund_rate.to_numpy()
    df.index = fed_fund_rate.index
    for series_id, params in indicators:
        # Get the data from FRED, convert to pandas DataFrame
        indicator = fred.get_series(series_id, **params)
        indicator = indicator.to_frame().set_axis([series_id], axis="columns")
        # fill in data with '0.0' that is presented as just '.'
        indicator[series_id] = ["0.0" if x == "." else x for x in indicator[series_id]]
        # turn the value into numeric
        indicator[series_id] = pd.to_numeric(indicator[series_id])
        indicator.index = pd.to_datetime(indicator.index).to_period("M")
        indicator = indicator.resample("M").interpolate()

        if series_id in ("PAYEMS", "T5YIE", "GDPC1"):  # align 1 lag
            indicator = indicator.shift(-1)[:-1]

        if series_id in ("MEDCPIM158SFRBCLE"):  # align 2 lag
            indicator = indicator.shift(-2)[:-2]
            indicator.rename(columns={"MEDCPIM158SFRBCLE": "MEDCPI"}, inplace=True)

        if series_id in ("EMRATIO"):  # align 3 lag
            indicator = indicator.shift(-3)[:-3]

        if series_id in ("LES1252881600Q"):  # align 5 lag
            indicator = indicator.shift(-5)[:-5]
            indicator.rename(columns={"LES1252881600Q": "MEDWAGES"}, inplace=True)

        # if series_id in ('NEWORDER'): # align 2 lead
        #     indicator = indicator.shift(2)[2:]

        # if series_id in ('JTS3000JOL'): # align 4 lead
        #     indicator = indicator.shift(4)[4:]

        if series_id in ("PERMIT", "AMTMNO", "DGORDER", "T10Y3M"):  # align 5 lead
            indicator = indicator.shift(5)[5:]
            # print(indicator)

        # join the dataframes together
        df = pd.concat([indicator, df], axis="columns")

    # DO DATA IMPUTATION FOR POSSIBLE NAN VALUES
    df = df.fillna(method="ffill")

    # remove outliers
    z_scores = zscore(df)
    abs_z_scores = np.abs(z_scores)
    threshold = 2.5
    filtered_entries = (abs_z_scores < threshold).all(axis=1)
    df = df[filtered_entries]

    return df


def download_data(docs, from_year):
    df = docs.get_contents(from_year)
    print("Shape of the downloaded data: ", df.shape)
    print("The first 5 rows of the data: \n", df.head())
    print("The last 5 rows of the data: \n", df.tail())
    docs.pickle_dump_df(filename=f"{batch_id}_{docs.content_type}" + ".pickle")


def download_fed_futures_historical():
    # Futures: full historical + forward data
    # download from barchart
    futures = pd.read_csv("../data/fed_futures_data/raw_data/historical-prices.csv")
    futures = futures[:-1]
    futures['Exp Date'] = pd.to_datetime(futures['Exp Date'])
    futures = futures.set_index("Exp Date")

    # options = webdriver.ChromeOptions() ;
    # options.add_argument("--disable-notifications")
    # prefs = {"download.default_directory" : "/Users/erica/Desktop/capstone/data_historical",
    #         'download.prompt_for_download': False,
    #         'download.directory_upgrade': True,
    #         'safebrowsing.enabled': False,
    #         'safebrowsing.disable_download_protection': True};
    # options.add_experimental_option("prefs",prefs);
    
    # driver = webdriver.Chrome(executable_path='./chromedriver', options=options)

    # driver.get('https://www.barchart.com/futures/quotes/ZQQ22/historical-prices?viewName=main&orderBy=percentChange1y&orderDir=desc&page=1');
    # driver.implicitly_wait(5)
    # popup = driver.find_element_by_xpath("""/html/body/div[4]/i""")
    # popup.click()

    # button = driver.find_element_by_xpath("""//*[@id="main-content-column"]/div/div[2]/div[2]/div[2]""")
    # button.click();

    # username = driver.find_element_by_xpath("""//*[@id="bc-login-form"]/div[1]/input""")
    # username.send_keys("fohigix903@stvbz.com")
    # password = driver.find_element_by_xpath("""//*[@id="login-form-password"]""")
    # password.send_keys("bestpassword")
    # login_button = driver.find_element_by_xpath("""//*[@id="bc-login-form"]/div[4]/button""")
    # login_button.click()

    # button = driver.find_element_by_xpath("""//*[@id="main-content-column"]/div/div[2]/div[2]/div[2]""")
    # button.click()
    # print("download")
    # time.sleep(10)
    # driver.close()
    
    return futures

def download_fomc_dates():
        page = requests.get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")
        soup = BeautifulSoup(page.content, 'html.parser')
        soupy = soup.find_all("div", class_="panel panel-default")
        result = []
        for s in soupy:
            year = s.find("div", class_ = "panel-heading").get_text()
            year = re.findall(r'\d+', year)[0]

            year_meetings = s.find_all("div", class_=["row fomc-meeting","fomc-meeting--shaded row fomc-meeting"])
            for meeting in year_meetings:
                meeting_month = meeting.find("div", class_="fomc-meeting__month").get_text()
                meeting_day = meeting.find("div", class_="fomc-meeting__date").get_text()

                if re.search(r"\(([^)]+)", meeting_day):
                    continue
                meeting_month = meeting_month.split("/")[0]
                meeting_day = meeting_day.split("-")[0]

                try: 
                    d = datetime.datetime.strptime(f"{meeting_day} {meeting_month} {year}", '%d %B %Y')
                except:
                    d = datetime.datetime.strptime(f"{meeting_day} {meeting_month} {year}", '%d %b %Y')
                result.append(d.strftime('%Y-%m-%d'))
        result.sort()
        df = pd.DataFrame(result)
        df.columns = ['Date']
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index("Date")

        return df


if __name__ == "__main__":
    pg_name = sys.argv[0]
    args = sys.argv[1:]
    content_type_all = ("statement", "minutes", "news", "all")

    if (len(args) != 1) and (len(args) != 2):
        print("Usage: ", pg_name)
        print("Please specify the first argument from ", content_type_all)
        print("You can add from_year (yyyy) as the second argument.")
        print("\n You specified: ", ",".join(args))
        sys.exit(1)

    if len(args) == 1:
        from_year = 1990
    else:
        from_year = int(args[1])

    content_type = args[0].lower()
    if content_type not in content_type_all:
        print("Usage: ", pg_name)
        print("Please specify the first argument from ", content_type_all)
        sys.exit(1)

    if (from_year < 1980) or (from_year > 2021):
        print("Usage: ", pg_name)
        print("Please specify the second argument between 1980 and 2020")
        sys.exit(1)

    if content_type == "all":
        fomc = FomcStatement()
        download_data(fomc, from_year)
        fomc = FomcMinutes()
        download_data(fomc, from_year)
        news = News()
        download_data(news, from_year)

    else:
        if content_type == "statement":
            docs = FomcStatement()
        elif content_type == "minutes":
            docs = FomcMinutes()
        else:  # News
            docs = News()

        download_data(docs, from_year)
