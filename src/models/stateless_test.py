import numpy as np
from src.models.baseline import NaiveNoChange
import tensorflow as tf

from src.models.stateless import LinearSingleStep
from src.test.list import is_equal


# Single Step
def test_LinearSingleStep():
    model = LinearSingleStep()
    model.compile(loss=tf.losses.MeanSquaredError(),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    inputs = np.asarray([[1, 1, 1, 1], [2, 4, 2, 2], [4, 6, 8, 10]])
    prediction = model(inputs)
    # This should yield a slope of 2 as (4-2)/(3-2) = 2
    assert is_equal(prediction, [6, 8, 14, 18])
