from src.helpers import get_processed_file
from src.logger import log
import matplotlib.pyplot as plt


# Produces a histogram of bet cycles to show how many bet cycles there usually are per race.
def get_statistics(file_path):
    df = get_processed_file(file_path)
    log('Calculating how many bet cycles exist per race')
    race_cycles = df.groupby(['refid']).size().reset_index(name='counts')
    print(f"Race betting cycles: "
          f"\n  - average: {race_cycles['counts'].mean()}"
          f"\n  - min: {race_cycles['counts'].min()}"
          f"\n  - max: {race_cycles['counts'].max()}"
          f"\n  - median: {race_cycles['counts'].median()}")
    race_cycles.hist(column='counts', bins=60)
    plt.ylabel('# of races')
    plt.xlabel('# of cycles per race')
    plt.title('Number of bet cycles in the races')
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    get_statistics('../../data/win_only_data.csv')
