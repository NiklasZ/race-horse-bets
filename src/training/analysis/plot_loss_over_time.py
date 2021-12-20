from datetime import datetime
from typing import List, Dict
import numpy as np
from matplotlib import pyplot as plt

from src.logger import log
from src.training.analysis.plot_prediction import get_models
from src.training.helpers import get_training_config
from src.training.prepare_data import get_normalised_data, get_training_data
import tensorflow as tf

# TODO put plotting utilities in shared file
markers = ['P', 's', 'p', 'X', '*', 'D']
colours = ['black', 'aqua', 'blue', 'brown', 'coral', 'goldenrod', 'green', 'indigo', 'lime', 'magenta', 'red',
           'purple', 'turquoise', 'yellow', 'sienna', 'khaki']


# Average losses for races that occurred on the same date.
def average_by_date(date_strings: List[str], losses: np.ndarray) -> [List[str], np.ndarray]:
    # {date_string, [model, loss_type]}
    loss_dict = {}
    for date, loss in zip(date_strings, losses):
        if date not in loss_dict:
            loss_dict[date] = []

        loss_dict[date].append(loss)

    averaged_losses = []
    averaged_ids = []
    for date_string in sorted(loss_dict):
        losses = np.asarray(loss_dict[date_string])
        averaged = np.mean(losses, axis=0)
        averaged_losses.append(averaged)
        averaged_ids.append(date_string)

    return averaged_ids, np.asarray(averaged_losses)


def plot_prediction_loss(names: List[str], models: List[tf.keras.Model], inputs: np.ndarray, labels: np.ndarray,
                         refids: List[str]):
    # Get the date prefix from refid
    date_strings = [d[:10] for d in refids]

    # Shape: [race, model, loss_type]
    losses = []
    # Get model predictions
    for race_input, race_label in zip(inputs, labels):
        model_losses = []
        for model in models:
            prediction = np.squeeze(model(race_input))
            mse_loss = tf.keras.metrics.mean_squared_error(race_label, prediction).numpy()[0]
            msa_calculator = tf.keras.metrics.MeanAbsoluteError()
            msa_calculator.update_state(race_label, prediction)
            msa_percentage_loss = msa_calculator.result().numpy()
            model_losses.append([mse_loss, msa_percentage_loss])
        losses.append(model_losses)

    log(f'Calculated losses for {len(models)} models')
    losses_arr = np.asarray(losses, dtype=object)
    averaged_dates, averaged_losses = average_by_date(date_strings, losses_arr)

    x_values = [datetime.strptime(d, "%Y-%m-%d").date() for d in averaged_dates]

    for i, colour in zip(range(averaged_losses.shape[1]), colours):
        model_name = names[i]
        mse_losses = averaged_losses[:, i, 0]
        # TODO MSA
        msa_losses = averaged_losses[:, i, 1]
        plt.plot(x_values, mse_losses, marker='.', linestyle='-', color=colour,
                 label=f'{model_name} - MSE')

    plt.ylabel('Error (Normalised)')
    plt.xlabel(f'Race date')
    plt.title(f'Model Error over all races from {date_strings[0]} to {date_strings[-1]}')
    plt.legend()
    plt.show()

    return


if __name__ == '__main__':
    data_path = '../../../data/2_prepared/2021-07-13T23-25-28'
    # model_path = '../../../data/3_trained/2021-09-03T07-59-25-new-quintuple-mlp'
    model_path = '../../../data/3_trained/2021-08-31T16-53-44-eternal-patience'
    config = get_training_config(f'{data_path}/config.json')
    n, m = get_models(model_path, ['NaiveNoChange', 'UnivariateLinear', 'UnivariateSingleMLPDelta'])

    # normalised_data = get_normalised_data(data_path, '2020-08')
    normalised_data = get_normalised_data(data_path)
    [train_series, validation_series, test_series, train_ids, validation_ids, test_ids] = normalised_data

    [train_inputs, train_labels, validation_inputs, validation_labels, test_inputs, test_labels] = get_training_data(
        data_path, config, normalised_data)

    log('Plotting training loss')
    plot_prediction_loss(n, m, train_inputs, train_labels, train_ids)

    log('Plotting validation loss')
    plot_prediction_loss(n, m, validation_inputs, validation_labels, validation_ids)

    log('Plotting test loss')
    plot_prediction_loss(n, m, test_inputs, test_labels, test_ids)
