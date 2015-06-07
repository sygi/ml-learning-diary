# [Probabilistic graphical models course](https://class.coursera.org/pgm-003/lecture)

## Status:
- [x] Introduction and Overview
- [x] Bayesian Network Fundamentals
- [x] Template Models (01.03.15)
- [x] Structured CPDs
- [x] Markov Network Fundamentals (15.03.15)
- [x] Representation Wrapup: Knowledge Engineering (24.03.15)
- [x] Inference: Variable Elimination (29.03.15)
- [x] Inference: Belief Propagation (21.04.15)
- [x] Inference: MAP Estimation, Part 1 (03.05.15)
- [x] Inference: MAP Estimation, Part 2
- [x] Inference: Sampling Methods (09.05.15)
- [x] Inference: Temporal Models and Wrap-up
- [x] Decision Theory (15.05.15)
- [x] Learning: Overview (19.05.15)
- [x] [Learning: Parameter Estimation in BNs](#learning-in-bns) (24.05.15)
- [x] [Learning: Parameter Estimation in MNs](#learning-in-mns) (06.06.15)
- [x] [Structure Learning](#structure) (07.06.15)
- [ ] Learning With Incomplete Data
- [ ] Learning: Wrapup
- [ ] Summary

## Learning in BNs:
Module contains definition of likelihood and the way of calculating it for bayesian networks.
Later, it introduces calculating joint (x,y) density through bayesian inference (posterior) and gives examples for bernoulli and dirichlet variables.

## Learning in MNs:
The videos talk about map estimation in Markov Networks and CFRs. It contains the derivation of log-linear model, the derivative of it and the proof that it is concave.
To calculate map in both (MRF and CRF) they use gradient ascent (or quasi-Newton method), but in CRF inference has to be run for each point separately, but for a smaller number of features (only output ones).

## Structure:
Module gives hints about learning the structure of a pgm using data. The process consists of two parts:
- defining scoring function for the structure
- finding the model that minimizes it

For the scoring function, there are two natural candidates:
 - likelihood score = the value of the likelihood the model has after choosing parameters to be MLE. This model is a bit bad, as it favors more complicated structures
- BIC score that is equal to likelihood - term depending on number of parameters. This formulation tries to reduce overfitting.
Furthermore, likelihood can be decomposed as sum of mutual informations of the variables and some entropies, independent of the structure.

Further, finding the best structure is discussed. For forests (where every node as at most one parent) the problem is easy and can be reduced to the problem of maximum spanning tree. For anything other, problem is NP-complete and heuristic approaches, using something like gradient descent are used. There are some techniques to avoid local minimas, as well as ones that are made to reduce the complexity of one step of search to O(Mn + nlogn), where n - number of nodes, M - number of datapoints.

