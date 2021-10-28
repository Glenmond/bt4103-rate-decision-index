import pandas
class Model():
    def __init__(self, country: str):
        self.country = country
        # handle the input for the data(API?, DB?), using country to get the country's data
        # is our data going to be in a pandas dataset?
        # TODO: data input into the model

    def preprocess(self):
        pass
        # TODO: handle data preprocessing
    
    def predict(self):
        pass
        # TODO: handle how prediction is made, using input data

    def create_index(self):
        pass
        # TODO: handle how an index is created using our predictions 