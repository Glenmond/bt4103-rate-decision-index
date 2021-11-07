from backtest import Backtest
from report import Report
from datetime import datetime

class Main():
    
    if __name__ == '__main__':
        print(f"Running Federal Funds Future Model")
        bt = Backtest()
        result = bt.predict()

        today = datetime.now().strftime("%y%m%d")
        result.to_csv(f"../data/fed_futures_data/result/{today}_fff_result.csv")

        report = Report(result)
        report.generate_prediction_graphs(save=True)

        print(f"Model Running completed!")
