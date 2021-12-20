import matplotlib.pyplot as plt
import numpy as np
import pandas
from tqdm import tqdm
from src.helpers import get_processed_file
from matplotlib.ticker import FuncFormatter

# TODO DOC
from src.logger import log


def millions(x, pos):
    return '%1.0fM' % (x * 1e-6)


def plot_average(series: pandas.Series):
    seconds = series.index
    amounts = series.array
    plt.plot(seconds, amounts, marker='.')
    # Disables scientific notation
    # plt.ticklabel_format(style='plain')

    formatter = FuncFormatter(millions)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(formatter)

    plt.ylabel('Average Amount Bet per Race (HKD)')
    plt.xlabel('Time (seconds) until betting ends')
    plt.title(f'Betting Volume before a Race')
    plt.show()


def calculate_average_over_time(df: pandas.DataFrame) -> pandas.Series:
    races = df['refid'].unique()
    cumulative_series = pandas.Series([])
    for refid in tqdm(races):
        race = df.loc[df['refid'] == refid]
        bets = race[['bet_amount_horse_nb_1', 'bet_amount_horse_nb_2', 'bet_amount_horse_nb_3', 'bet_amount_horse_nb_4',
                     'bet_amount_horse_nb_5', 'bet_amount_horse_nb_6', 'bet_amount_horse_nb_7', 'bet_amount_horse_nb_8',
                     'bet_amount_horse_nb_9', 'bet_amount_horse_nb_10', 'bet_amount_horse_nb_11',
                     'bet_amount_horse_nb_12',
                     'bet_amount_horse_nb_13', 'bet_amount_horse_nb_14']]
        bets_mean = bets.sum(axis=1)
        bet_times = race['_seconds'] - race['_seconds'].max()
        rounded_times = bet_times.round()
        combined = pandas.Series(bets_mean.array, index=rounded_times)
        cumulative_series = cumulative_series.add(combined, fill_value=0)
    averaged_by_race = cumulative_series.divide(len(races))
    return averaged_by_race


def get_percentage(bet_series: pandas.Series, position: int) -> float:
    return bet_series.array[position] / bet_series.array[-1] * 100


def get_volumes(bet_series: pandas.Series, time_steps=15):
    # - 1 to account for final position
    log(f'Average volume bet 60 minutes before: {get_percentage(bet_series, int(-3600 / time_steps - 1))}%')
    log(f'Average volume bet 30 minutes before: {get_percentage(bet_series, int(-1800 / time_steps - 1))}%')
    log(f'Average volume bet 10 minutes before: {get_percentage(bet_series, int(-600 / time_steps - 1))}%')
    log(f'Average volume bet 2 minutes before: {get_percentage(bet_series, int(-120 / time_steps - 1))}%')
    log(f'Average volume bet 30 seconds before: {get_percentage(bet_series, int(-30 / time_steps - 1))}%')


# Gets the average betting volume during the betting period.
# Indicates that most bets come in towards the end
def chart_betting_over_time(file_path: str):
    df = get_processed_file(file_path)
    bet_series = calculate_average_over_time(df)
    get_volumes(bet_series)
    plot_average(bet_series)


if __name__ == '__main__':
    chart_betting_over_time('../../data/1_cleaned/win_only_data-2021-07-06T21-57-03.csv')
