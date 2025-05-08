from abc import ABC, abstractmethod
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
    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score.
        """
        pass

    def _polynomial_features(self, X):
        """
            Generate polynomial features from the input features.
            Check the slides for hints on how to implement this one. 
            This method is used by the regression models and works
            for any degree polynomial. The returned value also contains
            a column of 1's at the start to account for the bias term.
            Parameters:
            X (array-like): Features of the data.

            Returns:
            X_poly (array-like): Polynomial features (extended).
        """
        # May be completely wrong
        # X_poly = np.c_(*[X**i for i in range (0, self.degree+1)])
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        features = [np.ones(X.shape[0])]  # bias term

        for deg in range(1, self.degree + 1):
            features.extend([X[:, j]**deg for j in range(X.shape[1])])

        X_poly = np.c_[*features]
        return X_poly

class RegressionModelNormalEquation(MachineLearningModel):
    """
    Class for regression models using the Normal Equation for polynomial regression.
    """

    def __init__(self, degree):
        """
        Initialize the model with the specified polynomial degree.

        Parameters:
        degree (int): Degree of the polynomial features.
        """
        self.beta = None
        self.degree = degree

    def fit(self, X, y):
        """
        Train the model using the given training data.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        Xe = self._polynomial_features(X)
        y = y.flatten()
        
        # May need handling for ininveritiblity (np.linalg.pinv)
        self.beta = np.linalg.inv(Xe.T @ Xe) @ Xe.T @ y

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        Xe = self._polynomial_features(X)
        predictions = Xe @ self.beta
        return predictions.flatten()

    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score (MSE).
        """
        y = y.flatten()
        y_pred = self.predict(X)
        
        score = np.mean((y - y_pred)**2)
        return score

class RegressionModelGradientDescent(MachineLearningModel):
    """
    Class for regression models using gradient descent optimization.
    """

    def __init__(self, degree, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the model with the specified parameters.

        Parameters:
        degree (int): Degree of the polynomial features.
        learning_rate (float): Learning rate for gradient descent.
        num_iterations (int): Number of iterations for gradient descent.
        """
        self.beta = None
        self.degree = degree
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def fit(self, X, y):
        """
        Train the model using the given training data.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        Xe = self._polynomial_features(X)
        y = y.flatten()
        self.beta = np.zeros(Xe.shape[1])
        n = Xe.shape[0]
        
        for j in range(0, self.num_iterations):
            gradient = -2 * Xe.T @ (Xe @ self.beta - y) / n
            self.beta = self.beta - self.learning_rate * gradient

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        Xe = self._polynomial_features(X)
        predictions = Xe @ self.beta
        return predictions.flatten()

    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score (MSE).
        """
        y = y.flatten()
        y_pred = self.predict(X)
        
        score = np.mean((y - y_pred)**2)
        return score

class LogisticRegression(MachineLearningModel):
    """
    Logistic Regression model using gradient descent optimization.
    """

    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the logistic regression model.

        Parameters:
        learning_rate (float): The learning rate for gradient descent.
        num_iterations (int): The number of iterations for gradient descent.
        """
        self.beta = None
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        y = y.flatten()
        self.beta = np.zeros(X.shape[1])
        n = X.shape[0]
        
        for j in range (0, self.num_iterations):
            gradient = X.T @ (self._sigmoid(X @ self.beta) - y) / n
            self.beta = self.beta - self.learning_rate * gradient

    def predict(self, X):
        """
        Make predictions using the trained logistic regression model.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted probabilities.
        """
        predictions = self._sigmoid(X @ self.beta)
        return predictions.flatten()

    def evaluate(self, X, y):
        """
        Evaluate the logistic regression model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score (e.g., accuracy).
        """
        y = y.flatten()
        y_pred = (self.predict(X) >= 0.5).astype(int)
        
        score = np.sum(y_pred == y) / y.shape[0]
        return score

    def _sigmoid(self, z):
        """
        Sigmoid function.

        Parameters:
        z (array-like): Input to the sigmoid function.

        Returns:
        result (array-like): Output of the sigmoid function.
        """
        return 1 / (1 + np.exp(-z))

    def _cost_function(self, X, y):
        """
        Compute the logistic regression cost function.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        cost (float): The logistic regression cost.
        """
        y = y.flatten()
        y_pred = np.clip(self.predict(X), 1e-15, 1 - 1e-15)
        n = X.shape[0]

        cost = -1/n * (y.T @ np.log(y_pred) + (1 - y).T @ np.log(1 - y_pred))
        return cost
    
class NonLinearLogisticRegression(MachineLearningModel):
    """
    Nonlinear Logistic Regression model using gradient descent optimization.
    It works for 2 features (when creating the variable interactions)
    """

    def __init__(self, degree=2, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the nonlinear logistic regression model.

        Parameters:
        degree (int): Degree of polynomial features.
        learning_rate (float): The learning rate for gradient descent.
        num_iterations (int): The number of iterations for gradient descent.
        """
        self.beta = None
        self.degree = degree
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def fit(self, X, y):
        """
        Train the nonlinear logistic regression model using gradient descent.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        X = self._mapFeature(X[:,0], X[:,1], self.degree)
        y = y.flatten()
        self.beta = np.zeros(X.shape[1])
        n = X.shape[0]
        
        for j in range (0, self.num_iterations):
            gradient = X.T @ (self._sigmoid(X @ self.beta) - y) / n
            self.beta = self.beta - self.learning_rate * gradient

    def predict(self, X):
        """
        Make predictions using the trained nonlinear logistic regression model.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted probabilities.
        """
        X = self._mapFeature(X[:,0], X[:,1], self.degree)
        predictions = self._sigmoid(X @ self.beta)
        return predictions.flatten()

    def evaluate(self, X, y):
        """
        Evaluate the nonlinear logistic regression model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        cost (float): The logistic regression cost.
        """
        y = y.flatten()
        y_pred = (self.predict(X) >= 0.5).astype(int)
        
        score = np.sum(y_pred == y) / y.shape[0]
        return score

    def _sigmoid(self, z):
        """
        Sigmoid function.

        Parameters:
        z (array-like): Input to the sigmoid function.

        Returns:
        result (array-like): Output of the sigmoid function.
        """
        return 1 / (1 + np.exp(-z))

    def _mapFeature(self, X1, X2, D):
        """
        Map the features to a higher-dimensional space using polynomial features.
        Check the slides to have hints on how to implement this function.
        Parameters:
        X1 (array-like): Feature 1.
        X2 (array-like): Feature 2.
        D (int): Degree of polynomial features.

        Returns:
        X_poly (array-like): Polynomial features.
        """
        features = []
        for i in range(D + 1):
            for j in range(i+1):
                feature = (X1**(i-j)) * (X2**j)
                features.append(feature)
        X_poly = np.c_(*features)
        return X_poly

    def _cost_function(self, X, y):
        """
        Compute the logistic regression cost function.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        cost (float): The logistic regression cost.
        """
        y = y.flatten()
        y_pred = np.clip(self.predict(X), 1e-15, 1 - 1e-15)
        n = X.shape[0]

        cost = -1/n * (y.T @ np.log(y_pred) + (1 - y).T @ np.log(1 - y_pred))
        return cost
