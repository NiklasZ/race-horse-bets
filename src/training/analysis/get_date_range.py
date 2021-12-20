from typing import List
from src.logger import log


def get_date_range(race_ids: List[str]):
    sorted_ids = race_ids.copy()
    sorted_ids.sort()
    if sorted_ids[0] != race_ids[0] or sorted_ids[-1] != race_ids[-1]:
        raise Exception('Expected date ids to be sorted.')

    log(f'Races range from {sorted_ids[0]} to {sorted_ids[-1]}')
