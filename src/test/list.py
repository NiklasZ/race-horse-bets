def is_equal(actual, expected) -> bool:
    return all([a == b for a, b in zip(actual, expected)])
