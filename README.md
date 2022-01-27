# Horse Race Betting

This is a project that explores to what extent we can predict the final betting odds of a Hong Kong horse race. 
In this type of betting system the odds are only known at the end as they are directly proportional to the total amount bet on each horse. 
For example, if players bet a total of $5000 on horse A and $1000 on horse B during the betting period, then the final odds are 5:1. 

### Why is predicting betting odds useful?
Because knowing the final odds beforehand will also tell you how much you can win based on the bets placed, which is useful to maximise returns for risk (win rate of the horse).
Suppose for example, you know that horse A has a 80% chance of winning  and horse B has a 20% chance. Then if you know what the final odds you can determine which bet has a higher expected return. For example if the final betting odds are 10:1 (horse A: horse B), then the expected return on betting $100 on either horse is:

Return_A = 0.8 * 100 * 0.1 = $8

Return_B = 0.2 * 100 * 10 = $200

Which makes it clear which bet on average, yields a better return. Of course, in reality most betting participants have some sense of which horses have winning potential and the odds are correlated with the bet pools. However, this is not perfect and predicting the bet pools beforehand helps here.

## Data used

The data for this project is not in the repository, but can be shared on demand (contact nz911174@gmail.com). It contains the betting data of 3163 races in Hong Kong from 09/2017 to 05/2021. Each race is a series of timestamps with corresponding bet pool values of up to 14 horses. Additionally, depending on the type of betting (place, win, quinella and omni), there is a corresponding bet pool.

## Example
Here is the development of bet amounts of horses for a win pool race id "2017-09-03-2622462", where the last values on the time axis determine the final odds:

![Alt text](/images/2021-07-contiguous-data/original-2017-09-03-2622462.png "Optional Title")

## Running it

This project uses Python 3.9.6. Install dependencies from the `Pipfile.lock` into your `pipenv`:
```bash
pipenv install
```

There are scripts for each step of the data analysis:
1. `src/preprocesser.ipynb` - checks race data and removes races with insufficient or corrupt data and generally simplifies the data for easier processing. From this we can plot the races and inspect them as shown above.
2. `src/generate_training_data.ipynb` - converts the races into chunks usable for machine learning.
3. `src/training/train.py` - trains models using the training data.
4. `src/training/hyper_tune.py` - tunes hyperparameters of models for even better results.

## Analysis
The prediction work isn't really done yet, but some basic linear models and standard multi-layer-perceptrons (MLPs) have been explored. These predict betting roughly 30 seconds into the future. The mean-squared error (MSE) of these models is shown below:

![Alt text](/images/2021-09-14-report/total_loss_using_only_last_year.png "Optional Title")

Here "NaiveNoChange" is the naive assumption that our betting odds to not change from our last known datapoint.
