from dataclasses import dataclass
from typing import List

import numpy as np
import pandas
from tqdm import tqdm
from src.helpers import generate_new_combination_indices
from src.logger import log


def load_prepared_data(file_path: str) -> pandas.DataFrame:
    df = pandas.read_csv(file_path, dtype={
        '_id': str,
        'refid': str,
        'bet_amount_horse_nb_1': float,
        'bet_amount_horse_nb_2': float,
        'bet_amount_horse_nb_3': float,
        'bet_amount_horse_nb_4': float,
        'bet_amount_horse_nb_5': float,
        'bet_amount_horse_nb_6': float,
        'bet_amount_horse_nb_7': float,
        'bet_amount_horse_nb_8': float,
        'bet_amount_horse_nb_9': float,
        'bet_amount_horse_nb_10': float,
        'bet_amount_horse_nb_11': float,
        'bet_amount_horse_nb_12': float,
        'bet_amount_horse_nb_13': float,
        'bet_amount_horse_nb_14': float,
    }, index_col=0)
    log(f"Read in training data of {df['refid'].nunique()} races, comprising {len(df.index)} rows")
    return df


def trim_irrelevant_rows(df: pandas.DataFrame) -> [np.ndarray, List[str]]:
    race_ids = df['refid'].unique()
    races = []
    for refid in tqdm(race_ids):
        race = df.loc[df['refid'] == refid]
        # This is slow, but picking or dropping are both slow.
        cleaned = race.drop(columns=['_id', 'refid'])
        races.append(cleaned.to_numpy())

    if len(race_ids) != len(races):
        raise Exception('Race id count does not match the # of races')

    as_numpy = np.asarray(races)
    log(f"Converted data into numpy array of shape {as_numpy.shape}")
    return [as_numpy, race_ids.tolist()]


# Convert data into an input (x) and label (y) set for evaluation.
# Will also omit intermediate steps based on how many it should predict into the future
def split_into_inputs_and_labels(to_split: np.ndarray, steps: int) -> [np.ndarray, np.ndarray]:
    inputs = to_split[:, :-steps, :]
    labels = np.expand_dims(to_split[:, -1, :], axis=1)
    return [inputs, labels]


# Inflate the dataset to a given factor with randomised indices
# As we have up to 14 horses, this means we can scale the dataset by a factor of up to
# 14! = 87178291200
def inflate_data(to_inflate: np.ndarray, indices: np.ndarray) -> np.ndarray:
    new_data = []
    for d in to_inflate:
        for i in indices:
            new_data.append(d.T[i].T)

    return np.asarray(new_data)


def get_training_data(folder_path: str, config: dict) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                          np.ndarray, List[str]]:
    data = load_prepared_data(f'{folder_path}/data.csv')
    [training_data, race_ids] = trim_irrelevant_rows(data)

    # Split into training, validation and testing set
    n = len(training_data)
    training_set = training_data[0:int(n * 0.7)]
    validation_set = training_data[int(n * 0.7):int(n * 0.9)]
    test_set = training_data[int(n * 0.9):]
    test_ids = race_ids[int(n * 0.9):]

    norm_training = training_set / training_set.max()
    norm_validation = validation_set / training_set.max()
    norm_test = test_set / training_set.max()

    if 'data_inflation_factor' not in config:
        config['data_inflation_factor'] = 1

    if 'cycles_into_the_future' not in config:
        config['cycles_into_the_future'] = 2

    combination_indices = np.asarray(
        generate_new_combination_indices(config['random_seed'], 14, config['data_inflation_factor']))

    train_inputs, train_labels = split_into_inputs_and_labels(inflate_data(norm_training, combination_indices),
                                                              config['cycles_into_the_future'])
    validation_inputs, validation_labels = split_into_inputs_and_labels(
        inflate_data(norm_validation, combination_indices),
        config['cycles_into_the_future'])
    test_inputs, test_labels = split_into_inputs_and_labels(norm_test, config['cycles_into_the_future'])

    return [train_inputs, train_labels, validation_inputs, validation_labels, test_inputs, test_labels,
            test_ids]
