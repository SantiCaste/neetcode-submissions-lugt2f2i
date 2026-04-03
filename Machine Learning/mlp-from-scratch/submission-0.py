import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        # pass

        # option 1:
        i = x
        for layer in range(0, len(weights) - 1):
            z = i @ weights[layer] + biases[layer]
            i = np.maximum(0, z)

        # for the final layer:
        z = i @ weights[-1] + biases[-1]

        return np.round(z, 5)

