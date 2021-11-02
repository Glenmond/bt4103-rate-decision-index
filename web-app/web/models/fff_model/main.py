from web.models.fff_model.backtest import Backtest
from web.models.fff_model.report import Report

class Main():
    
    #if __name__ == '__main__':
    def run_main(self):
        print(f"Running Federal Funds Future Model")
        bt = Backtest()
        result = bt.predict()

        report = Report(result)
        report.generate_prediction_graphs(save=True)

        print(f"Model Running completed!")
