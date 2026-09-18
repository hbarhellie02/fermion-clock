# Numerical simulation code for fermion clock

> Repository accompanying the paper:
> **Exponential gain in clock precision using quantum correlated ticks**  
> *Florian Meier, Yuri Minoguchi, Gianmichele Blasi, Géraldine Haack, and Marcus Huber*
>
> **Paper:** arXiv:[2601.10785](https://arxiv.org/abs/2601.10785) [quant-ph] (2026)


# Overview

Below is a summary of the main Python files included in the repository.
This code is published for the purpose of independently verifying the results of the above mentioned paper, and for follow-up research.

| File | Purpose |
|------|---------|
| `main.py` | Commands to run the plotting functions used for the visualizations of the paper (except the Anderson localization plots). |
| `visualizations.py` | Contains plotting functions. Note that the figures in the paper have been asthetically enhanced using Inkscape.
| `transmission_stability.py` | Code for simulating disorder and visualization.
| `transmission_clock.py` | Various functions to calculate relevant quantities of the clock: Transmission function, Finite-time noise D(t) and asymptotic noise D/J, steady-state covariance matrix
| `mc_covariance_cluster.py` | Code for stochastic Monte Carlo simulation of the tick times, reasonably optimized for performance.

The code was last tested using Python 3.10.11, with package versions as detailed in the `requirements.txt` file. Some of the plotting functions require a working LaTeX installation such as TeX Live.

# Contact / Authorship

This code has been authored by Florian Meier with input from all authors.

Paper: **Exponential gain in clock precision using quantum correlated ticks**
arXiv:[2601.10785](https://arxiv.org/abs/2601.10785) [quant-ph] (2026)

For questions concerning the work and the code please contact:

- **Florian Meier**
- **Email:** [florianmeier256@gmail.com](mailto:florianmeier256@gmail.com)

If you use this repository in your work, please cite the accompanying paper.