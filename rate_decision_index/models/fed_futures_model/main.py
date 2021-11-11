from models.fed_futures_model.backtest import Backtest
from models.fed_futures_model.report import Report
from datetime import datetime


    
def run_main(path = "../data/fed_futures_data"):
    print(f"Running Federal Funds Future Model")
    bt = Backtest(path)
    result, pred_values, probs = bt.predict()

    today = datetime.now().strftime("%y%m%d")
    result.to_csv(f"{path}/result/{today}_fff_result.csv")
    pred_values.to_csv(f"{path}/result/{today}_fff_preds.csv")
    probs.to_csv(f"{path}/result/{today}_fff_raw_probs.csv")

    result.to_csv(f"{path}/latest/fff_result.csv")
    pred_values.to_csv(f"{path}/latest/fff_preds.csv")
    probs.to_csv(f"{path}/latest/fff_raw_probs.csv")

    report = Report(result, path)
    report.generate_prediction_graphs(save=True)

    print(f"Model Running completed!")

if __name__ == '__main__':
    run_main()
