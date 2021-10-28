# this is where we do our main analysis using our models and data
import numpy

from data import fetch_data
from models import MacroModel, MacroData

if __name__ == '__main__':
    # Get the data
    data = fetch_data()

    macro_data = MacroData(data)
    macro_model = MacroModel(macro_data)

    macro_model.fit_data()

    macro_model.assess_val_set_performance()
    macro_model.assess_test_set_performance()
