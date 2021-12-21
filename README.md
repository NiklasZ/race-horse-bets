# Horse Race Betting

This is a hobby project that explores to what extent we can predict the final betting odds of a Hong Kong horse race. 
In this type of betting system the odds are only known at the end as they are directly proportional to the total amount bet on each horse. 
For example, if players bet a total of $5000 on horse A and $1000 on horse B during the betting period, then the final odds are 5:1. 

### Why is predicting betting odds useful?
Because knowing the final odds beforehand will also tell you how much you can win based on the bets placed, which is useful to maximise returns for risk (win rate of the horse).
Suppose for example, you know that horse A has a 80% chance of winning  and horse B has a 20% chance. Then if you know what the final odds you can determine which bet has a higher expected return. For example if the final betting odds are 10:1 (horse A: horse B), then the expected return on betting $100 on either horse is:

Return_A = 0.8 * 100 * 0.1 = $8

Return_B = 0.2 * 100 * 10 = $200

Which makes it clear which bet on average, yields a better return. Of course, in reality most betting participants have some sense of which horses have winning potential and the odds correlated with he bet pools. However, this is not perfect and predicting the bet pools beforehand helps here.

## Data used

