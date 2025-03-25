# BurgersJoint
Solving Burgers' equation with spectral methods, finite difference methods, and physics-informed neural networks. Written in Julia.

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