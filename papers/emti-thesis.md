[Emti thesis](https://circle.ubc.ca/bitstream/handle/2429/43640/ubc_2013_spring_khan_mohammad.pdf?sequence=3)
=============================================================================================================

[x] [chapter 1](#introduction) (10.06.15)
[x] [chapter 2](#Discrete-Data LGMs) (11.06.15)

## Introduction
The chapter first defines Latent Gaussian Models - general probabilistic graphical models, assuming gaussian distribution of latent factors. It shows the example models that fit this class and problems that can be solved using these. 

## Discrete-data LGMs
Second chapter is a review of existing approaches for learning LGMs.

### Non-Bayesian approaches
The one approach that is presented here is something like EM, but finding argmaxes for both latent variables (MAP) and parameters. I would expect this would give a gad result due to MAP inaccuracy, but apparently it runs quite fine, but tends to overfit and careful choice of regularization is required.

### Sampling methods
These are differnet variations on the Metropolis-Hastings algorithm, where we estimate posterior distribution using sampling. The various methods exist to improve accuracy/mixing time.

### Deterministic methods
There are also some methods that are good only in some particular situations. I didn't understand most of the derivations (and why they were done), but there are two methods presented: Laplace method, that estimates posterior integral using mode of some function and doesn't work well in models with multiple parameters due to expected grid search and Expectation-Propagation algorithm, which I didn't understand.
