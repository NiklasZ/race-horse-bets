import math
import os
from datetime import datetime

import keras_tuner as kt
import tensorflow as tf

from src.logger import log
from src.models.univariate_double_mlp import UnivariateDoubleMLP, compile_model as double_mlp_compile
from src.models.univariate_quintuple_mlp_delta import UnivariateQuintupleMLPDelta, \
    compile_model as quintuple_mlp_delta_compile
from src.models.univariate_single_mlp_delta import UnivariateSingleMLPDelta, compile_model as single_mlp_delta_compile
from src.models.univariate_triple_mlp import UnivariateTripleMLP, compile_model as triple_mlp_compile
from src.models.univariate_triple_mlp_delta import UnivariateTripleMLPDelta, compile_model as triple_mlp_delta_compile
from src.training.helpers import get_training_config
from src.training.prepare_data import get_training_data, get_normalised_data

if len(tf.config.list_physical_devices('GPU')) == 0:
    raise Exception('No GPU found')

hyperparameter_compilers = {
    UnivariateDoubleMLP().__class__.__name__: double_mlp_compile,
    UnivariateTripleMLP().__class__.__name__: triple_mlp_compile,
    UnivariateSingleMLPDelta().__class__.__name__: single_mlp_delta_compile,
    UnivariateTripleMLPDelta().__class__.__name__: triple_mlp_delta_compile,
    UnivariateQuintupleMLPDelta().__class__.__name__: quintuple_mlp_delta_compile,
}


def compile_and_hypertune(model_builder, model_name: str, training_data, output_folder: str, patience=5, max_epochs=20):
    [train_inputs, train_labels, validation_inputs, validation_labels, *_] = training_data

    training_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))
    validation_dataset = tf.data.Dataset.from_tensor_slices((validation_inputs, validation_labels))

    factor = 3
    hyperband_iterations = 3
    tuner = kt.Hyperband(model_builder,
                         objective='val_loss',
                         max_epochs=max_epochs,
                         factor=factor,
                         hyperband_iterations=hyperband_iterations,
                         directory=output_folder,
                         project_name=model_name)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    tuner.search(training_dataset, validation_data=validation_dataset,
                 callbacks=[early_stopping])

    tuner.results_summary()


def main(existing_tuning_folder=None):
    current_model = UnivariateQuintupleMLPDelta()
    name = current_model.__class__.__name__

    log(f'Tuning Model {name}')
    # 100 cycle data
    path = '../../data/2_prepared/2021-07-13T23-25-28'
    # 25 cycle data
    # path = '../../data/2_prepared/2021-08-13T16-54-57'

    config = get_training_config(f'{path}/config.json')
    normalised_data = get_normalised_data(path, '2020-08')
    training_data = get_training_data(path, config, normalised_data)

    model_builder = hyperparameter_compilers.get(name)

    if model_builder is None:
        raise Exception(f"Can't find compilation function for model {name}")

    date_string = existing_tuning_folder if existing_tuning_folder is not None else datetime.now().strftime(
        '%Y-%m-%dT%H-%M-%S')
    output_folder = f'../../data/4_tuned/{date_string}'

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    compile_and_hypertune(model_builder, name, training_data, output_folder, patience=5, max_epochs=50)


if __name__ == '__main__':
    main()
