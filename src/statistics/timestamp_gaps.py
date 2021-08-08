import pandas
import pandas as pd
from src.helpers import get_processed_file
from src.logger import log
import matplotlib.pyplot as plt
from tqdm import tqdm


# TODO filter unrestricted version by year
# Measures how much deviation there is between cycle updates.
def get_statistics(file_path):
    df = get_processed_file(file_path)
    log('Calculating how much of a time gap there is between each cycle in the data.')
    cycle_differences = pd.Series([])
    races = df['refid'].unique()
    for refid in tqdm(races):
        race = df.loc[df['refid'] == refid]
        time_strings = race[['asattimestr']].squeeze()
        timestamps = pandas.to_timedelta(time_strings)
        cycle_differences = cycle_differences.append(timestamps.diff().dropna())
    as_seconds = cycle_differences.dt.total_seconds()

    print(f"Cycle differences in seconds: "
          f"\n  - average: {as_seconds.mean()}"
          f"\n  - min: {as_seconds.min()}"
          f"\n  - max: {as_seconds.max()}"
          f"\n  - median: {as_seconds.median()}"
          f"\n  - standard deviation: {as_seconds.std()}")

    as_seconds.hist(bins=60)
    plt.ylabel('# of cycles')
    plt.xlabel('time per gap (seconds)')
    plt.gcf().subplots_adjust(left=0.15)
    plt.title('Distribution of cycle time gaps')
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    get_statistics('../../data/1_cleaned/win_only_data-2021-06-17T00-04-45.csv')
    get_statistics('../../data/1_cleaned/win_only_data-2021-07-06T21-57-03.csv')
