import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
from utils.parse.data_read import load_mat_variable

import matplotlib.pyplot as plt

from instances.ridge_shuffle_decoding import run_decoding_with_shuffle


def example_run(
    x_path="data/test/Fdff.mat",
    y_path="data/test/rotationalVelocity.mat",
):
    run_decoding_with_shuffle(x_path, y_path)


if __name__ == "__main__":
    example_run()