import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # Compute Y_hat = X @ W (matrix multiplication)
        # X is (n, 3), weights is (3,) -> result is (n,) predictions
        # Return np.round(result, 5)
        # pass
        Y_hat = X @ weights
        return np.round(Y_hat, 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute MSE = mean((predictions - truth)^2)
        # Use np.mean() and np.square()
        # Return round(result, 5)
        # pass
        mse = np.mean(np.power(model_prediction - ground_truth, 2))
        return round(mse, 5)

        # this doesn't work, i'm not sure why:
        # mse = 1 / len(model_prediction) * np.pow(model_prediction - ground_truth, 2)
        #return np.round(mse, 5)