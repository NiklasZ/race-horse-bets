import numpy as np
from src.models.baseline import NaiveNoChange
import tensorflow as tf
from src.test.list import is_equal


# Single Step
def test_NaiveSingleStep():
    baseline = NaiveNoChange()
    baseline.compile(loss=tf.losses.MeanSquaredError(),
                     metrics=[tf.metrics.MeanAbsoluteError()])
    inputs = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]])
    prediction = baseline(inputs)
    assert is_equal(prediction, [5, 6, 7, 8])
