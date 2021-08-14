import json
import os
from datetime import datetime
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from src.logger import log
from src.models.baseline import NaiveNoChange
from src.models.multistep_dense import MultistepDense
from src.models.single_layer import SingleLayerDense
from src.models.stateless import LinearSingleStep
from src.models.univariate_double_mlp import UnivariateDoubleMLP
from src.models.univariate_linear import UnivariateLinear
from src.models.univariate_sextuple_mlp import UnivariateSextupleMLP
from src.models.univariate_single_mlp import UnivariateSingleMLP
from src.models.univariate_triple_mlp import UnivariateTripleMLP
from src.training.prepare_data import get_training_data

if len(tf.config.list_physical_devices('GPU')) == 0:
    raise Exception('No GPU found')


# TODO
# This is a good tutorial: https://www.tensorflow.org/tutorials/structured_data/time_series

def get_training_config(file_path: str) -> dict:
    with open(file_path) as f:
        return json.load(f)


def compile_and_fit(model: tf.keras.Model, train_inputs: np.ndarray, train_labels: np.ndarray,
                    training_dataset: tf.data.Dataset,
                    validation_dataset: tf.data.Dataset, patience=5, max_epochs=20):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
    learning_rate = model.tuned_learning_rate if hasattr(model, 'tuned_learning_rate') else 0.001

    # TODO consider cutting down the features from 100 to 50 or 25
    # TODO experiment with the MAE rather than MSE.
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                  metrics=[tf.metrics.MeanAbsoluteError()]) # tf.metrics.MeanSquaredError()

    if train_labels[0].shape != model(train_inputs[0]).shape:
        raise Exception('The training inputs and model outputs have different shapes.'
                        f'Training label:{train_labels[0].shape}'
                        f'Model output: {model(train_inputs[0]).shape}')

    model.summary()

    trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])

    # Allows us to skip models that don't actually have anything to train
    if trainable_params > 0:
        history = model.fit(training_dataset, epochs=max_epochs,
                            validation_data=validation_dataset,
                            callbacks=[early_stopping]
                            )
    else:
        history = None

    return history


def train_models(models: List[tf.keras.Model], config: dict, output_folder: str, training_data):
    [train_inputs, train_labels, validation_inputs, validation_labels, test_inputs, test_labels,
     test_ids] = training_data

    training_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))
    validation_dataset = tf.data.Dataset.from_tensor_slices((validation_inputs, validation_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_inputs, test_labels))

    training_performances = {}
    validation_performances = {}
    test_performances = {}

    for m in models:
        config['training_epochs'] = 100
        config['patience'] = 5

        training_history = compile_and_fit(m, train_inputs, train_labels, training_dataset, validation_dataset,
                                           max_epochs=config['training_epochs'], patience=config['patience'])

        name = m.__class__.__name__

        training_performances[name] = m.evaluate(training_dataset)
        validation_performances[name] = m.evaluate(validation_dataset)
        test_performances[name] = m.evaluate(test_dataset)
        log(f'{name}: validation performance:{validation_performances[name]}')
        log(f'{name}: test performance:{test_performances[name]}')

        if training_history is not None:
            history_path = f'{output_folder}/{name}_training.json'
            with open(history_path, 'w') as outfile:
                json.dump(training_history.history, outfile, indent=4)
            m.save(f'{output_folder}/{name}.model')

    updated_config_path = f'{output_folder}/config.json'
    with open(updated_config_path, 'w') as outfile:
        json.dump(config, outfile, indent=4)

    performances = {'training_performances': training_performances, 'validation_performances': validation_performances,
                    'testing_performances': test_performances}
    performances_path = f'{output_folder}/performances.json'
    with open(performances_path, 'w') as outfile:
        json.dump(performances, outfile, indent=4)


def main():
    log('Starting model training...')
    path = '../data/2_prepared/2021-07-13T23-25-28'
    config = get_training_config(f'{path}/config.json')

    training_data = get_training_data(path, config)

    all_models = [NaiveNoChange(), LinearSingleStep(), UnivariateLinear(), SingleLayerDense(), MultistepDense(),
                  UnivariateSingleMLP(), UnivariateSextupleMLP()]
    interesting_models = [UnivariateTripleMLP(), UnivariateDoubleMLP(), NaiveNoChange(), UnivariateLinear(),
                          UnivariateSingleMLP()]
    current_models = [UnivariateSextupleMLP(), UnivariateLinear(), UnivariateSingleMLP()]

    models = current_models

    date_string = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    output_folder = f'../data/3_trained/{date_string}'

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    train_models(models, config, output_folder, training_data)
    log('Done.')


if __name__ == '__main__':
    main()
