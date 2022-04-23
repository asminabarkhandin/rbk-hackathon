import pickle
import pandas as pd

class MyModel():
    
    def __init__(self, path):
        file = open(path, 'rb')
        self.model = pickle.load(file)
        file.close()
        
    def pred_danet(self, iin):
        X = self.load_X(iin)
        X = X.drop('ID', axis=1)
        pred = self.model.predict(X)
        if pred[0] == 1:
            return 'Одобрено'
        else:
            return 'Не одобрено'

    def pred_monthly(self, iin):
        X = self.load_X(iin)
        X = X.drop('ID', axis=1)
        pred = self.model.predict(X)
        return round(pred[0])


    def load_X(self, iin):
        df = pd.read_csv('hackathon_db.csv', dtype={'ID': str})
        X = df[df['ID'] == iin]
        return X
