import os
from typing import List
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from src.logger import log
from src.models.baseline import NaiveNoChange
from src.training.helpers import get_training_config
from src.training.prepare_data import get_training_data, get_normalised_data


def get_model_directories(path: str):
    r = os.listdir(path)
    return [entry for entry in os.listdir(path) if os.path.isdir(f'{path}/{entry}') and '_model' in entry]


def get_models(path: str, filter_by: List[str] = None) -> [List[str], List[tf.keras.Model]]:
    matches = get_model_directories(path)
    matches.sort()
    log(f'Found {len(matches)} models to plot')
    if len(matches) == 0:
        raise Exception('No models to plot')

    model_names = [folder.replace('_model', '') for folder in matches]
    model_instances = [tf.keras.models.load_model(f'{path}/{match}') for match in matches]

    # Insert the Naive baseline for comparison
    naive_no_change = NaiveNoChange()
    model_names.insert(0, naive_no_change.__class__.__name__)
    naive_no_change.compile(loss=tf.losses.MeanSquaredError(),
                            optimizer=tf.optimizers.Adam(learning_rate=0.01),
                            metrics=[tf.metrics.MeanAbsoluteError()])
    model_instances.insert(0, naive_no_change)

    # FIXME refactor so model and name are tuples in one list.
    if filter_by:
        filtered_names = []
        filtered_instances = []
        for name, model in zip(model_names, model_instances):
            if name in filter_by:
                filtered_names.append(name)
                filtered_instances.append(model)
        model_names = filtered_names
        model_instances = filtered_instances

    return model_names, model_instances


markers = ['P', 's', 'p', 'X', '*', 'D']
colours = ['black', 'aqua', 'blue', 'brown', 'coral', 'goldenrod', 'green', 'indigo', 'lime', 'magenta', 'red',
           'purple', 'turquoise', 'silver', 'sienna', 'khaki']


def plot_predictions(names: List[str], models: List[tf.keras.Model], race_id: str, race_input, race_label, race_series):
    [input_cycles, input_pools] = race_input.shape

    # Plot the existing time series and future
    for input_column, series_column, colour in (zip(race_input.T, race_series.T, colours)):
        plt.plot(range(0, input_cycles), input_column, marker='.', color=colour)
        future_values = np.insert(series_column[len(input_column):], 0, input_column[-1])
        plt.plot(range(input_cycles - 1, input_cycles - 1 + len(future_values)), future_values, marker='.',
                 color=colour,
                 alpha=0.3, linewidth=2)

    # Plot model predictions
    for name, model, marker in zip(names, models, markers):
        predictions = np.squeeze(model(race_input))
        label_set = False
        for p, colour in zip(predictions, colours):
            label = None if label_set else f'{name}'
            plt.plot(len(race_series) - 1, p, marker, linestyle='', color=colour,
                     label=label)
            label_set = True

    plt.ylabel('Bet Pool - normalised')
    plt.xlabel(f'Cycles (30 seconds each)')
    plt.title(f"Race {race_id} - Model Predictions")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    data_path = '../../../data/2_prepared/2021-07-13T23-25-28'
    model_path = '../../../data/3_trained/2021-09-02T20-14-10-only-new-season-2'
    config = get_training_config(f'{data_path}/config.json')
    n, m = get_models(model_path)

    normalised_data = get_normalised_data(data_path, '2020-08')
    [train_series, validation_series, test_series, training_ids, validation_ids, test_ids] = normalised_data

    [train_inputs, train_labels, validation_inputs, validation_labels, test_inputs, test_labels] = get_training_data(
        data_path, config, normalised_data)

    for idx in range(len(test_series)):
        plot_predictions(n, m, test_ids[idx], test_inputs[idx], test_labels[idx],
                         test_series[idx])
