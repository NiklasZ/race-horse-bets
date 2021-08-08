from src.helpers import get_processed_file
from src.logger import log
import numpy as np


def get_statistics(file_path):
    df = get_processed_file(file_path)
    log('Calculating statistics of race data...')
    print(f"Number of races: {df['refid'].nunique()}")
    race_cycles = df.groupby(['refid']).size()
    print(f"Race betting cycles: "
          f"\n  - average: {np.mean(race_cycles)}"
          f"\n  - median: {np.median(race_cycles)}"
          f"\n  - min: {np.min(race_cycles)}"
          f"\n  - max: {np.max(race_cycles)}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    get_statistics('../../data/1_cleaned/win_only_data-2021-07-06T21-57-03.csv')
