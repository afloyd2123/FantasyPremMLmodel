import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class PlayerPerformanceModel:

    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def train(self, X_train, y_train):
        """
        Train the Random Forest model.
        """
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        """
        Predict player performance.
        """
        return self.model.predict(X_test)
        
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model's RMSE.
        """
        y_pred = self.predict(X_test)
        return mean_squared_error(y_test, y_pred, squared=False)

def train_and_evaluate(data, target_col):
    """
    Split the data, train the model, and evaluate performance.
    """
    X = data.drop(columns=[target_col])
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = PlayerPerformanceModel()
    model.train(X_train, y_train)
    
    rmse = model.evaluate(X_test, y_test)
    return model, rmse
