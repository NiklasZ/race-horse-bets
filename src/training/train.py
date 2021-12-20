import json
import os
from datetime import datetime
from typing import List
import numpy as np
import tensorflow as tf
from src.logger import log
from src.models.baseline import NaiveNoChange
from src.models.multistep_dense import MultistepDense
from src.models.single_layer import SingleLayerDense
from src.models.stateless import LinearSingleStep
from src.models.univariate_double_mlp import UnivariateDoubleMLP
from src.models.univariate_linear import UnivariateLinear
from src.models.univariate_quintuple_mlp_delta import UnivariateQuintupleMLPDelta
from src.models.univariate_sextuple_mlp import UnivariateSextupleMLP
from src.models.univariate_single_mlp import UnivariateSingleMLP
from src.models.univariate_single_mlp_delta import UnivariateSingleMLPDelta
from src.models.univariate_triple_mlp import UnivariateTripleMLP
from src.models.univariate_triple_mlp_delta import UnivariateTripleMLPDelta
from src.training.helpers import get_training_config
from src.training.prepare_data import get_training_data, get_normalised_data
import shutil

if len(tf.config.list_physical_devices('GPU')) == 0:
    raise Exception('No GPU found')


# TODO
# This is a good tutorial: https://www.tensorflow.org/tutorials/structured_data/time_series

def remove_dir(mydir: str):
    for f in os.listdir(mydir):
        if not f.endswith(".bak"):
            continue
        os.remove(os.path.join(mydir, f))


def compile_and_fit(model: tf.keras.Model, train_inputs: np.ndarray, train_labels: np.ndarray,
                    training_dataset: tf.data.Dataset,
                    validation_dataset: tf.data.Dataset, checkpoint_folder: str, patience=5, max_epochs=20):
    learning_rate = model.tuned_learning_rate if hasattr(model, 'tuned_learning_rate') else 1e-2
    patience = model.tuned_patience if hasattr(model, 'tuned_patience') else patience

    # Controls when to stop training if the validation loss doesn't improve
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
    # Take checkpoint of best model for evaluation
    checkpoint_path = f'{checkpoint_folder}/model_checkpoint'
    if not os.path.exists(checkpoint_folder):
        os.mkdir(checkpoint_folder)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    monitor='val_loss',
                                                    verbose=1,
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    mode='min')

    # TODO consider cutting down the features from 100 to 50 or 25
    # TODO experiment with the MAE rather than MSE.
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                  metrics=[tf.metrics.MeanAbsoluteError()])  # tf.metrics.MeanSquaredError()

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
                            callbacks=[early_stopping, checkpoint]
                            )
        # Load best result for evaluation
        model.load_weights(checkpoint_path)
    else:
        history = None

    shutil.rmtree(checkpoint_folder)

    return history


def train_models(models: List[tf.keras.Model], config: dict, output_folder: str, training_data):
    [train_inputs, train_labels, validation_inputs, validation_labels, test_inputs, test_labels] = training_data

    training_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))
    validation_dataset = tf.data.Dataset.from_tensor_slices((validation_inputs, validation_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_inputs, test_labels))

    training_performances = {}
    validation_performances = {}
    test_performances = {}

    for m in models:
        config['training_epochs'] = 300
        config['patience'] = 50

        checkpoint_folder = f'{output_folder}/checkpoint'
        training_history = compile_and_fit(m, train_inputs, train_labels, training_dataset, validation_dataset,
                                           checkpoint_folder,
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
            m.save(f'{output_folder}/{name}_model')

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
    run_alias = '-new-quintuple-mlp'
    config = get_training_config(f'{path}/config.json')

    normalised_data = get_normalised_data(path, '2020-08')
    training_data = get_training_data(path, config, normalised_data)

    all_models = [NaiveNoChange(), LinearSingleStep(), UnivariateLinear(), SingleLayerDense(), MultistepDense(),
                  UnivariateSingleMLP(), UnivariateSextupleMLP()]
    interesting_models = [UnivariateSingleMLPDelta(), UnivariateTripleMLP(), UnivariateDoubleMLP(), NaiveNoChange(),
                          UnivariateLinear(),
                          UnivariateSingleMLP(), UnivariateTripleMLPDelta(), UnivariateQuintupleMLPDelta()]
    current_models = [NaiveNoChange(), UnivariateQuintupleMLPDelta(), UnivariateTripleMLPDelta(),
                      UnivariateSingleMLPDelta(),
                      UnivariateLinear()]

    models = current_models

    date_string = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    output_folder = f'../data/3_trained/{date_string}{run_alias}'

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    train_models(models, config, output_folder, training_data)
    log('Done.')


if __name__ == '__main__':
    main()
