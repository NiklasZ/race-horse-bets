from src.logger import log
from src.training.analysis.count_empty_races import count_empty_races
from src.training.analysis.get_date_range import get_date_range
from src.training.prepare_data import get_normalised_data


def compare_data():
    pass


if __name__ == '__main__':
    data_path = '../../../data/2_prepared/2021-07-13T23-25-28'
    [train_series, validation_series, test_series, training_ids, validation_ids, test_ids] = get_normalised_data(
        data_path)

    log('Training Data:')
    count_empty_races(train_series)
    get_date_range(training_ids)

    log('Validation Data:')
    count_empty_races(validation_series)
    get_date_range(validation_ids)

    log('Test Data:')
    count_empty_races(test_series)
    get_date_range(test_ids)
