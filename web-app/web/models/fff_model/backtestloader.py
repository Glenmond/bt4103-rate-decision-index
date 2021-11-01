import pandas as pd
import os
from datetime import datetime,timedelta
import re
import yfinance as yf
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
import time

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
        futures = pd.read_csv("web/fff_model/data/historical-prices.csv")
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
    
    def load_meeting_futures_data(self, meeting_date,period='max'):
        year = meeting_date.strftime('%y')
        code = self.contract_codes[meeting_date.strftime("%B")]
        ticker = f"ZQ{code}{year}.CBT"
        df = yf.Ticker(ticker).history(period=period)

        return df
    
    def get_fomc_dates(self):
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
                    d = datetime.strptime(f"{meeting_day} {meeting_month} {year}", '%d %B %Y')
                except:
                    d = datetime.strptime(f"{meeting_day} {meeting_month} {year}", '%d %b %Y')
                result.append(d.strftime('%Y-%m-%d'))
        result.sort()
        df = pd.DataFrame(result)
        df.columns = ['Date']
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index("Date")

        if not os.path.exists("data"):
            os.mkdir("data")
        df.to_csv("data/fomc_dates.csv")
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
            
    