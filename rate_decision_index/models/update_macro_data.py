# Updates the pickle files with the latest data
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression

try:
    from macro_model.macroeconomics_model import MacroData, MacroModel
except ModuleNotFoundError: 
    from .macro_model.macroeconomics_model import MacroData, MacroModel
    
try:
    from extract import fetch_data
except ModuleNotFoundError:
    from .extract import fetch_data

def update_saved_data(path_to_folder = "./data/macroeconomic_indicators_data/", path_to_HD_folder = './data/sentiment_data/historical/'):
    data = fetch_data()

    # Save the data as a pickle
    data.to_pickle(path_to_folder + 'macro_data_pickle', protocol = 4)

    # Preprocess the data, add in the HD sentiment index and use it to fit the model
    macro_data = MacroData(data, path_to_HD_pickle=path_to_HD_folder)

    # fit the data on a base Linear Regression model to get the best coefficient for sensitivity analysis. 
    reg = LinearRegression().fit(macro_data.X_train, macro_data.y_train)
    best_coef_val = reg.coef_[0][-1]
    
    macro_model = MacroModel(macro_data, shift_coef = best_coef_val)
    macro_model.fit_data()

    # Save the model as a pickle
    with open(path_to_folder + 'macro_model_pickle', 'wb') as files:
        pickle.dump(macro_model, files, protocol = 4)

    # Get the training data and save as pickle
    X_train = macro_data.X_train.copy().append(macro_data.X_val).sort_index()
    y_train = macro_data.y_train.copy().append(macro_data.y_val).sort_index()

    X_train.to_pickle(path_to_folder + 'macro_X_train_pickle', protocol=4)
    y_train.to_pickle(path_to_folder + 'macro_y_train_pickle', protocol=4)

    # Get testing data and save as pickle
    X_test = macro_data.X_test
    y_test = macro_data.y_test

    X_test.to_pickle(path_to_folder+'macro_X_test_pickle', protocol=4)
    y_test.to_pickle(path_to_folder+'macro_y_test_pickle', protocol=4)

    # Get model performance on train and test and save as pickle
    _, y_perf = macro_model.predict(X_train)
    _, y_pred = macro_model.predict(X_test)
    y_perf= pd.DataFrame(y_perf,index=X_train.index.strftime('%Y-%m'), columns=['Federal Funds Rate'])
    y_pred = pd.DataFrame(y_pred,index=X_test.index.strftime('%Y-%m'), columns=['Federal Funds Rate'])

    y_perf.to_pickle(path_to_folder + 'macro_train_pred_pickle', protocol=4)
    y_pred.to_pickle(path_to_folder + 'macro_test_pred_pickle', protocol=4)

if __name__ == '__main__':
    update_saved_data()