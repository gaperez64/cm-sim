# Compartmental Model Simulator

This is a simple forward simulator of stochastic compartmental models meant to
be used for comparison of Monte Carlo methods to approximate values of
interest.

# TODO
The architecture, data generation, and training setup seem OK for now.
Sometimes, training yields good models with (validation and test) errors
in the order of 10e-2 for small SIR instances.

What's left to do?
*[] Hyperparameter tuning
*[] Use the neural stochastic matrix to compute the expected number of
  infectious people at the moment of peak (observed empirically); to compute
  the expected end of the epidemic (and compare against estimator)
*[] Think about alternatives to mean absolute error as loss function

Point 2 is the main open problem now. How does one evaluate products of the
matrix if we have a succinct representation thereof?
