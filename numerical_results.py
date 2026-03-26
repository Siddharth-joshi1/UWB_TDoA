# numerical_results.py

import numpy as np


def compute_metrics(errors):

    valid = errors[np.isfinite(errors)]

    mean_error = np.mean(valid)
    rmse = np.sqrt(np.mean(valid**2))
    std = np.std(valid)

    p90 = np.percentile(valid, 90)
    p95 = np.percentile(valid, 95)

    max_error = np.max(valid)

    failure_rate = 1 - (len(valid) / len(errors))

    metrics = {
        "Mean Error (m)": mean_error,
        "RMSE (m)": rmse,
        "Std Dev (m)": std,
        "90% Error (m)": p90,
        "95% Error (m)": p95,
        "Max Error (m)": max_error,
        "Failure Rate": failure_rate
    }

    return metrics


def print_metrics(metrics):

    print("\n===== Numerical Results =====")

    for k, v in metrics.items():

        print(f"{k:20s}: {v:.4f}")

    print("=============================\n")
def save_metrics(metrics, filename="results.txt"):

    with open(filename, "w") as f:

        f.write("Numerical Results\n")

        for k, v in metrics.items():

            f.write(f"{k}: {v}\n")

            
import matplotlib.pyplot as plt


def plot_cdf(errors):

    valid = errors[np.isfinite(errors)]

    sorted_errors = np.sort(valid)

    cdf = np.arange(len(sorted_errors)) / len(sorted_errors)

    plt.figure()

    plt.plot(sorted_errors, cdf)

    plt.xlabel("Localization Error (m)")
    plt.ylabel("CDF")

    plt.title("Localization Error CDF")

    plt.grid()

    plt.show()