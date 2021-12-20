from typing import List, Dict

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


def trim_irrelevant_rows(df: pandas.DataFrame, skip_rows_older_than=None) -> [np.ndarray, List[str]]:
    race_ids = df['refid'].unique()
    chosen_race_ids = []
    races = []

    if skip_rows_older_than:
        log(f'Omitting races with refids older than {skip_rows_older_than}')

    for refid in tqdm(race_ids):
        if skip_rows_older_than and refid < skip_rows_older_than:
            continue

        race = df.loc[df['refid'] == refid]
        # This is slow, but picking or dropping are both slow.
        cleaned = race.drop(columns=['_id', 'refid'])
        races.append(cleaned.to_numpy())
        chosen_race_ids.append(refid)

    # if len(race_ids) != len(races):
    #     raise Exception('Race id count does not match the # of races')

    as_numpy = np.asarray(races)
    log(f"Converted data into numpy array of shape {as_numpy.shape}")
    return [as_numpy, chosen_race_ids]


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


def get_normalised_data(folder_path: str, skip_older_than=None) -> [np.ndarray, np.ndarray, np.ndarray, List[str]]:
    data = load_prepared_data(f'{folder_path}/data.csv')
    [training_data, race_ids] = trim_irrelevant_rows(data, skip_older_than)

    # Split into training, validation and testing set
    n = len(training_data)
    training_set = training_data[0:int(n * 0.7)]
    validation_set = training_data[int(n * 0.7):int(n * 0.9)]
    test_set = training_data[int(n * 0.9):]

    training_ids = race_ids[0:int(n * 0.7)]
    validation_ids = race_ids[int(n * 0.7):int(n * 0.9)]
    test_ids = race_ids[int(n * 0.9):]

    norm_training = training_set / training_set.max()
    norm_validation = validation_set / training_set.max()
    norm_test = test_set / training_set.max()

    return [norm_training, norm_validation, norm_test, training_ids, validation_ids, test_ids]


def get_training_data(folder_path: str, config: dict, normalised_data=None) -> [np.ndarray, np.ndarray, np.ndarray,
                                                                                np.ndarray, np.ndarray,
                                                                                np.ndarray]:
    normalised_data = normalised_data if normalised_data is not None else get_normalised_data(folder_path, config)
    norm_training, norm_validation, norm_test, *_ = normalised_data

    if 'data_inflation_factor' not in config:
        config['data_inflation_factor'] = 1

    if 'cycles_into_the_future' not in config:
        config['cycles_into_the_future'] = 2

    if config['data_inflation_factor'] == 1:
        combination_indices = np.asarray([range(14)])
    else:
        combination_indices = np.asarray(
            generate_new_combination_indices(config['random_seed'], 14, config['data_inflation_factor']))

    train_inputs, train_labels = split_into_inputs_and_labels(inflate_data(norm_training, combination_indices),
                                                              config['cycles_into_the_future'])
    validation_inputs, validation_labels = split_into_inputs_and_labels(
        inflate_data(norm_validation, combination_indices),
        config['cycles_into_the_future'])
    test_inputs, test_labels = split_into_inputs_and_labels(norm_test, config['cycles_into_the_future'])

    return [train_inputs, train_labels, validation_inputs, validation_labels, test_inputs, test_labels]
