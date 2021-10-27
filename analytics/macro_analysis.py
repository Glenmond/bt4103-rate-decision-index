# this is where we do our main analysis using our models and data
import numpy

from data import fetch_data
from models import MacroModel

if __name__ == '__main__':
    # Get the data
    data = fetch_data()


    macro_model = MacroModel(data)
    macro_model.preprocess()
    
    macro_model.fit_data()
    print(macro_model.y_test)

    macro_model.assess_test_set_performance()
