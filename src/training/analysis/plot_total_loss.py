from typing import List
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from src.logger import log
from src.training.analysis.plot_prediction import get_models
from src.training.helpers import get_training_config
from src.training.prepare_data import get_normalised_data, get_training_data

# TODO doc
# TODO put plotting utilities in shared file
markers = ['P', 's', 'p', 'X', '*', 'D']
colours = ['black', 'aqua', 'blue', 'brown', 'coral', 'goldenrod', 'green', 'indigo', 'lime', 'magenta', 'red',
           'purple', 'turquoise', 'yellow', 'sienna', 'khaki']


def plot_prediction_loss(names: List[str], models: List[tf.keras.Model], all_data, refids: List[str]):
    [train_inputs, train_labels, validation_inputs, validation_labels, test_inputs, test_labels] = all_data

    date_strings = [d[:10] for d in refids]

    # Shape: [model, dataset, [mse loss, msa loss]]
    losses = []
    # Get model predictions
    for model in models:
        train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))
        validation_dataset = tf.data.Dataset.from_tensor_slices((validation_inputs, validation_labels))
        test_dataset = tf.data.Dataset.from_tensor_slices((test_inputs, test_labels))

        train_loss = model.evaluate(train_dataset)
        validation_loss = model.evaluate(validation_dataset)
        test_loss = model.evaluate(test_dataset)

        losses.append([train_loss, validation_loss, test_loss])

    log(f'Calculated losses for {len(models)} models')
    losses_arr = np.asarray(losses, dtype=object)

    plt.figure(figsize=(18, 9))

    x = np.arange(len(models))
    width = 0.2
    plt.bar(x - width, losses_arr[:, 0, 0], width, label='Training Error')
    plt.bar(x, losses_arr[:, 1, 0], width, label='Validation Error')
    plt.bar(x + width, losses_arr[:, 2, 0], width, label='Test Error')

    # plt.bar(names, mse_losses, color=colours[:len(names)], width=0.4)
    plt.xticks(x, names)
    plt.ylabel('Error (Normalised)')
    plt.title(f'Average Error over all races from {date_strings[0]} to {date_strings[-1]}')
    plt.legend()
    plt.show()

    return


if __name__ == '__main__':
    data_path = '../../../data/2_prepared/2021-07-13T23-25-28'
    model_path = '../../../data/3_trained/2021-09-02T20-14-10-only-new-season-2'
    # model_path = '../../../data/3_trained/2021-08-31T16-53-44-eternal-patience'
    config = get_training_config(f'{data_path}/config.json')
    n, m = get_models(model_path, ['NaiveNoChange', 'UnivariateLinear', 'UnivariateSingleMLPDelta'])

    normalised_data = get_normalised_data(data_path, '2020-08')
    # normalised_data = get_normalised_data(data_path)
    [train_series, validation_series, test_series, train_ids, validation_ids, test_ids] = normalised_data

    training_data = get_training_data(data_path, config, normalised_data)
    all_ids = train_ids + validation_ids + test_ids

    log('Plotting total loss')
    plot_prediction_loss(n, m, training_data, all_ids)
