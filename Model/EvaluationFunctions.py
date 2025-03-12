import tensorflow as tf
import numpy as np

@tf.function
def evaluate_model(model, test_dataset, sample_limit=1000):
    """
    Evaluate a model on a test dataset and return metrics and samples.
        This method is quite slow due to the nature of the dataset iteration.
        Recommended to use model.evaluate and model.predict for faster evaluation.
    """
    # Defining Metrics
    mse_metric = tf.keras.metrics.MeanSquaredError()
    mae_metric = tf.keras.metrics.MeanAbsoluteError()
    mape_metric = tf.keras.metrics.MeanAbsolutePercentageError()

    # We'll use TensorArray to collect samples without growing a Python list in a loop.
    # Here, we preallocate space for a maximum of 'sample_limit' batches (or you can decide on a max number of samples).
    sample_preds = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    sample_trues = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    sample_count = tf.constant(0)

    # Iterate over the dataset
    for inputs, true_labels in test_dataset:
        # Compute predictions (make sure model is defined in your context)
        preds = model(inputs, training=False)
        # Update metric state
        mse_metric.update_state(true_labels, preds)
        mae_metric.update_state(true_labels, preds)
        mape_metric.update_state(true_labels, preds)

        # Collect sample predictions if we haven't reached our limit.
        # For example, collect the first batch from each iteration until reaching sample_limit batches.
        if sample_count < sample_limit:
            sample_preds = sample_preds.write(sample_count, preds)
            sample_trues = sample_trues.write(sample_count, true_labels)
            sample_count += 1

    # Gather metric results and convert collected samples to tensors.
    mse_value = mse_metric.result()
    mae_value = mae_metric.result()
    mape_value = mape_metric.result()
    metrics = {'MSE': mse_value.results().numpy(),
               'MAE': mae_value.results().numpy(),
               'MAPE': mape_value.results().numpy()}

    collected_preds = sample_preds.concat().numpy()
    collected_trues = sample_trues.concat().numpy()

    return metrics, collected_preds, collected_trues

import matplotlib.pyplot as plt

def PlotTrueVsPredicted(sampled_trues, sampled_predictions, graph_name):
    """
    Plot the true values against the predicted values.

    Parameters
    ----------
    sampled_trues : numpy.ndarray
        The true values.
    sampled_predictions : numpy.ndarray
        The predicted values.

    """
    # Scatter plot of predictions vs actual alignment scores
    plt.figure(figsize=(12, 6))
    plt.scatter(sampled_trues, sampled_predictions, alpha=0.05)
    plt.plot([0, 1], [0, 1], 'r--')  # Line for reference
    plt.title(f'{graph_name} Predictions vs Actual Alignment Scores')
    plt.xlabel('Actual Alignment Scores')
    plt.ylabel('Predicted Alignment Scores')
    plt.show()

def PlotDifferenceHistogram(sampled_trues, sampled_predictions, graph_name):
    # Histogram of the differences
    differences = sampled_trues - sampled_predictions.flatten()
    plt.figure(figsize=(12, 6))
    plt.hist(differences, bins=50, alpha=0.75)
    plt.title(f'{graph_name} Histogram of Differences (Actual - Predicted)')
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    plt.show()

def PlotHeatMapScatter(sampled_trues, sampled_predictions, graph_name, method='hist2d'):
    plt.figure(figsize=(12, 6))

    # Ensure predictions are flattened
    sampled_predictions = sampled_predictions.flatten()

    # 2D histogram approach
    if method == 'hist2d':
        hist = plt.hist2d(sampled_trues, sampled_predictions, bins=50, cmap='viridis', cmin=0, cmax=350)
        plt.colorbar(label='Count')

    # Hexagonal binning approach
    elif method == 'hexbin':
        hb = plt.hexbin(sampled_trues, sampled_predictions, gridsize=500, cmap='viridis', mincnt=None)
        plt.colorbar(hb, label='Count')

    # Kernel density approach using scatter with color mapping
    elif method == 'kde':
        from scipy.stats import gaussian_kde

        # Calculate the point density
        xy = np.vstack([sampled_trues, sampled_predictions])
        z = gaussian_kde(xy)(xy)

        # Sort the points by density, so that the densest points are plotted last
        idx = z.argsort()
        x, y, z = sampled_trues[idx], sampled_predictions[idx], z[idx]

        sc = plt.scatter(x, y, c=z, s=10, cmap='viridis')
        plt.colorbar(sc, label='Density')

    # Add reference line
    plt.plot([0, 1], [0, 1], 'r--')

    plt.title(f'{graph_name} Heat Map of Predictions vs Actual Alignment Scores')
    plt.xlabel('Actual Alignment Scores')
    plt.ylabel('Predicted Alignment Scores')
    plt.show()