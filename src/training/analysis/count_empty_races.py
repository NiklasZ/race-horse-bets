import numpy as np
from src.logger import log


def count_empty_races(races: np.ndarray):
    count = 0
    cycle_series_count = races.shape[0] * races.shape[2]
    for race in races:
        bet_cycles = race.T
        for cycle_series in bet_cycles:
            if cycle_series[0] == 0 and cycle_series[-1] == 0:
                count += 1
    log(f'Found {count} empty bet pools ({round(count / cycle_series_count * 100)}% of all)')

