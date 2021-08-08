import matplotlib.pyplot as plt
import numpy as np
import pandas

from src.helpers import get_processed_file


def render_race(df: pandas.DataFrame, refid: str):
    race = df.loc[df['refid'] == refid]
    bets = race[['bet_amount_horse_nb_1', 'bet_amount_horse_nb_2', 'bet_amount_horse_nb_3', 'bet_amount_horse_nb_4',
                 'bet_amount_horse_nb_5', 'bet_amount_horse_nb_6', 'bet_amount_horse_nb_7', 'bet_amount_horse_nb_8',
                 'bet_amount_horse_nb_9', 'bet_amount_horse_nb_10', 'bet_amount_horse_nb_11',
                 'bet_amount_horse_nb_12',
                 'bet_amount_horse_nb_13', 'bet_amount_horse_nb_14']]

    timestamps = pandas.to_timedelta(race['asattimestr']).dt.total_seconds().to_numpy()
    matrix = bets.to_numpy()

    for count, column in enumerate(matrix.T):
        plt.plot(timestamps, np.asarray(column), label=f'bet pool {count}', marker='.')

    plt.ylabel('Bet Pool (HKD)')
    plt.xlabel('Time (seconds)')
    plt.title(f'Race {refid}')

    plt.show()


def chart_race(file_path: str, specific_refid: str = None):
    df = get_processed_file(file_path)

    if specific_refid is not None:
        render_race(df, specific_refid)
    else:
        races = df['refid'].unique()
        for refid in races:
            render_race(df, refid)


if __name__ == '__main__':
    # chart_race('../../data/win_only_data-2021-07-05T21-59-44.csv', '2018-01-21-2625211')
    # Synthetic
    chart_race('../../data/1_cleaned/win_only_data-2021-07-06T21-57-03.csv', '2017-09-03-2622462')
    # Original
    chart_race('../../data/1_cleaned/win_only_data-2021-06-22T20-58-40.csv', '2017-09-03-2622462')

    # chart_race('../../data/win_only_data-2021-07-06T21-57-03.csv')
