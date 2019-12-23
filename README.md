# SNIP-pruning

Report from ICLR Reproducibility Challenge.
Reproduced paper is SNIP: SINGLE-SHOT NETWORK PRUNING BASED ON CONNECTION SENSITIVITY (https://openreview.net/forum?id=B1VZqjAcYX).

There is .pdf file with report and source code of reproduction available. 
Link to the issue: https://github.com/reproducibility-challenge/iclr_2019/issues/130



## SNIP: Single-shot pruning

One can describe neural network’s loss sensitivity on zeroing a weight by its absolute change. We will call it the exact salience from now. Intuitively, if this sensitivity is small, then removing the parameter shouldn’t prevent neural network from learning, since it didn’t affect the loss function anyway. Although, one usually wants to remove more than one parameter and since they are dependent on each other, in order get the least loss affecting set of η parameters to prune, one needs to check every possible combination of η parameters. It requires this many forward passes in a network. Since this is computationally impossible, we are assuming that influence of weight on ∆L is independent from every other weight. This is the first of two assumptions we must make.

Since computing the exact salience for every weight separately is also computationally expensive – it requires m forward passes to compute (for m being number of parameters in the network) – we will assume that the exact salience corresponds to loss change for infinitesimally small ε as the argument difference. This second assumption is the one that describes the SNIP method the fullest. With it we can define a salience of a weight as the derivative of the loss with respect to the indicator value equal to 1. Such salience calculation can be done in modern frameworks very efficiently, for all weights at once. Salience could be then standardized to get parameters’ importances in percentages. After evaluating the saliences, one can set the indicator to 0 for selected connections and get the pruned network.