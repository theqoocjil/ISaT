import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


class KnnService():
    """
    Args:
        df: Initializing DataFrame
        feature: The main feature of the selection
    """
    
    def __init__(self, df: pd.DataFrame, feature: str):
        self._df = df
        self._knn = KNeighborsClassifier()
        if feature in self._df.columns:
            self._feature = feature
            self._y_train = df[feature]
        else:
            raise ValueError("Missing features in input data")
        self._scaler = StandardScaler()
        self._best_model = None
        self._best_params = None
        self._best_score = None
        self._is_trained = False

    def model_training(self, n_range: range):
        """
        Args:
            n_range: The range of the number of neighbors
        """
        X_train = self._df.drop(self._feature, axis=1)
        X_train_scaled = self._scaler.fit_transform(X_train)
        param_grid = {
            'n_neighbors': n_range,
            'metric': ["euclidean", "manhattan", "cosine", "chebyshev"],
        }
        grid_search = GridSearchCV(
            self._knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train_scaled, self._y_train)
        self._best_model = grid_search.best_estimator_
        self._best_params = grid_search.best_params_
        self._best_score = grid_search.best_score_
        self._is_trained = True

    def predict_data(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame for testing
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Добавить проверку сходимости фреймов
        if self._feature in df.columns:
            X_test = df.drop(self._feature, axis=1)
        else:
            X_test = df
        X_test_scaled = self._scaler.transform(X_test)
        predictions = self._best_model.predict(X_test_scaled)
        
        return predictions