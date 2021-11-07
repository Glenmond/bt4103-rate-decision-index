from backtest import Backtest
from report import Report

class Main():
    
    if __name__ == '__main__':
        print(f"Running Federal Funds Future Model")
        bt = Backtest()
        result = bt.predict()

        report = Report(result)
        report.generate_prediction_graphs(save=True)

        print(f"Model Running completed!")
