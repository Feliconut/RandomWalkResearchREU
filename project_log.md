# Project Log

## 2022-11-08

Work done on brownian motion test.
Study note entry for Hölder Exponent test: see `study note 04`. However, the proof contains an error: The scale invariance property is wrongly stated. To be fixed in future.
- [ ] Fix proof of Hölder Exponent test.

## 2022-09-26

*Theoretical background:*
- [x] Review mathematical detail for Autocorrelation function 
- [x] Understand Lipchitz and Hölder continuity

*Improvements on BM test:*
- [x] Implement and run test for Hölder continuity (see `study note 4` for theoretical details)
- [x] Implement Scale Invariance Test (Choose a small section and see if the distributions are identical up to proper scaling) (K-S Test on Empirical Distributions)

*Application of BM test:*
- [x] Run BM test on RWs with different step distribution
    - [x] Uniform
    - [x] Normal
    - [x] Exponential
    - [x] Cauchy


## 2022-09-23

- [ ] Vectorize the random number generation,
- [ ] Bit-wise storage of generated works, vectorize the query, piecewise summary.

Other ideas:
- Random seed for step generation
- Hashmap for path generation


## 2022-09-12

Tasks due before next meeting:

- [x] (possibly) vectorize the path generating code, make it faster
- [ ] Learn about K-S test and Chi-squared test, and implement in the code
- [x] Tune the experiment parameters (trial length, number of trials) and look for trends
- [x] Read about Brownian Motion and understand its defining properties
- [x] Design tests to determine if the data is Brownian Motion
- [x] Implement these tests and apply to SSRW
