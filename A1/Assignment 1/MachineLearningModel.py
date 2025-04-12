from abc import ABC, abstractmethod
from collections import Counter
import numpy as np

class MachineLearningModel(ABC):
    """
    Abstract base class for machine learning models.
    """

    @abstractmethod
    def fit(self, X, y):
        """
        Train the model using the given training data.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        pass

    @abstractmethod
    def evaluate(self, y_true, y_predicted):
        """
        Evaluate the model on the given data.

        Parameters:
        y_true (array-like): True target variable of the data.
        y_predicted (array-like): Predicted target variable of the data.

        Returns:
        score (float): Evaluation score.
        """
        pass
    
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2, axis=1))

class KNNRegressionModel(MachineLearningModel):
    """
    Class for KNN regression model.
    """

    def __init__(self, k):
        """
        Initialize the model with the specified instructions.

        Parameters:
        k (int): Number of neighbors.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Train the model using the given training data.
        In this case, the training data is stored for later use in the prediction step.
        The model does not need to learn anything from the training data, as KNN is a lazy learner.
        The training data is stored in the class instance for later use in the prediction step.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)

    def predict(self, X):
        """
        Make predictions on new data.
        The predictions are made by averaging the target variable of the k nearest neighbors.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        #--- Write your code here ---#
        predictions = []
        for x in X:
            distances = self._euclidean_distance(x, self.X_train)
            k_nearest_points = np.argsort(distances)[:self.k]
            predictions.append(np.mean(self.y_train[k_nearest_points]))
        
        return np.array(predictions)

    def evaluate(self, y_true, y_predicted):
        """
        Evaluate the model on the given data.
        You must implement this method to calculate the Mean Squared Error (MSE) between the true and predicted values.
        The MSE is calculated as the average of the squared differences between the true and predicted values.        

        Parameters:
        y_true (array-like): True target variable of the data.
        y_predicted (array-like): Predicted target variable of the data.

        Returns:
        score (float): Evaluation score.
        """
        return np.mean((y_true - y_predicted)**2)
        
    def _euclidean_distance(self, x1, x2):
        return super()._euclidean_distance(x1, x2)

class KNNClassificationModel(MachineLearningModel):
    """
    Class for KNN classification model.
    """

    def __init__(self, k):
        """
        Initialize the model with the specified instructions.

        Parameters:
        k (int): Number of neighbors.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Train the model using the given training data.
        In this case, the training data is stored for later use in the prediction step.
        The model does not need to learn anything from the training data, as KNN is a lazy learner.
        The training data is stored in the class instance for later use in the prediction step.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)

    def predict(self, X):
        """
        Make predictions on new data.
        The predictions are made by taking the mode (majority) of the target variable of the k nearest neighbors.
        
        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        #--- Write your code here ---#
        predictions = []
        for x in X:
            distances = self._euclidean_distance(x, self.X_train)
            k_nearest_points = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_nearest_points]
            prediction = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(prediction)
        return np.array(predictions)

    def evaluate(self, y_true, y_predicted):
        """
        Evaluate the model on the given data.
        You must implement this method to calculate the total number of correct predictions only.
        Do not use any other evaluation metric.

        Parameters:
        y_true (array-like): True target variable of the data.
        y_predicted (array-like): Predicted target variable of the data.

        Returns:
        score (float): Evaluation score.
        """
        return float(np.sum(y_true==y_predicted))
        
    def _euclidean_distance(self, x1, x2):
        """
        Calculates the euclidean distance between given points.
        
        Parameters:
        x1 (array-like): Point 1.
        x2 (array-like): Point 2.
        
        Returns:
        distance (float): The euclidean distance between the points.
        """
        return super()._euclidean_distance(x1, x2)
