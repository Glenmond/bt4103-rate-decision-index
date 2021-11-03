from abc import abstractmethod

class Model():
    @abstractmethod
    def preprocess(self):
        pass
        # TODO: handle data preprocessing

    @abstractmethod
    def fit_model(self):
        pass
    
    @abstractmethod
    def predict(self, data):
        pass
        # TODO: handle how prediction is made, using input data

    '''@abstractmethod
    def create_index(self):
        pass
        # TODO: handle how an index is created using our predictions''' 