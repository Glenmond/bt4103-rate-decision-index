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