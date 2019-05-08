# SNIP-pruning
Reproduction and analysis of SNIP paper


One can describe neural network’s loss sensitivity on zeroing a weight w j as s b j = |L(w j )−L(0)|. We will call it the exact
salience from now on. Intuitively, if this sensitivity is small, then removing the parameter shouldn’t prevent neural
network from learning, since it didn’t affect the loss function anyway. Although, one usually wants to remove more
than one parameter and, since they are dependent on each other, in order get the least loss affecting set of η parameters
to prune, one needs to check every possible combination of η parameters, thus make this many forward passes in a
network. Since this is computationally impossible, we are assuming that influence of w j on ∆L = |L(w j ) − L(0)| is
independent from every other weight than w j and this is the first of two assumptions we make.
Since computing s b j for every weight w j separately is also computationally expensive – it requires m forward passes
to compute (for m being number of parameters in the network) – we will assume that |L(w j ) − L(0)| corresponds to
|L(w j ) − L(w j − ε)| for infinitesimally small ε.
This second assumption is the one that describes the SNIP method. With it we can define a salience s j of a weight
∂L(w j ·c j )
for c j being an indicator variable equal to 1. Such salience calculation can be done in modern
w j as s j =
∂c j
frameworks very efficiently, for all weights at once. Salience can be then standardized to show parameters’ importances
in percentages. After evaluating the saliences, one can set the indicator c j to 0 for selected connections.
