import datetime
import sys

from dataloader import DataLoader
from datapreprocessor import DataPreprocessor
from backtest import Backtest


if __name__ == '__main__':
    args = sys.argv[1:]
    from_year = int(args[0])

    print(f"===== Running Hawkish-Dovish Index Model =====".title())
    batch_id = datetime.date.today().strftime("%y%m%d")
    dataloader = DataLoader(from_year)

    # Preprocessing
    datapreprecessor = DataPreprocessor(dataloader.data, batch_id)

    bt = Backtest(datapreprecessor.data, from_year)
    bt.predict()

    print(f"===== Modelling Process Completed =====".title())