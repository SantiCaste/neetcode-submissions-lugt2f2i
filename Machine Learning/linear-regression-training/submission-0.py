import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # For each iteration:
        #   1. Compute predictions with get_model_prediction(X, weights)
        #   2. For each weight index j, compute gradient with get_derivative()
        #   3. Update: weights[j] -= learning_rate * gradient
        # Return np.round(final_weights, 5)
        # pass

        # option 1:
        weights = initial_weights
        for it in range(0, num_iterations):
            # predict
            predictions = self.get_model_prediction(X, weights)

            # get derviates and train for each weight
            for w in range(len(weights)):
                # get derivative
                derivate = self.get_derivative(predictions, Y, len(Y), X, w)

                # train
                weights[w] -= self.learning_rate * derivate
            #derivates = self.get_derivative(predictions, Y, len(Y), X)
            #derivates = [self.get_derivative(predictions, Y, len(Y), X, w) for w in range(0, len(weights))]

            #weights -= self.learning_rate * derivates
        
        return np.round(weights, 5)