# BurgersJoint Documentation

Welcome to the documentation for BurgersJoint. This project provides various methods to solve Burgers' equation as formulated in
    Basdevant et al. (1986), "Spectral and Finite Difference Solutions of the Burgers Equation"
    (Basdevant1986).

## Overview

BurgersJoint is a Julia-based project that compares Fourier Galerkin, Fourier pseudospectral, Chebyshev Tau, finite difference, and PINN methods for solving Burgers’ equation.

## Modules

- `BurgersEquation`: Defines the PDE, initial conditions, and analytical solutions.
- `FourierSpectral`: Implements the Fourier Galerkin solver.

## Examples

```julia
using BurgersJoint
x = range(-1, 1, length=100)
t = 0.5
u_analytical = BurgersEquation.analytical_solution.(x, t)
```