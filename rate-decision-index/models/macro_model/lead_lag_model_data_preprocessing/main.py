import leadlagmodel as llm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quandl
from sklearn.preprocessing import MinMaxScaler

quandl.ApiConfig.api_key = "kVFwRskyFgKCs3HURnYV"

llmodel = llm.LeadLagModel()
results = llmodel.full_report
print(results)
results.to_csv('leadlaganalysis.csv')

