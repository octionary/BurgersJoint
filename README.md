# BurgersJoint

This project provides various methods to solve the 1D Burgers equation:
```math
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}
```
where $u(t, x)$ is the velocity field, $\nu$ is the kinematic viscosity, and $t$ and $x$ are time and space variables, respectively. The domain is defined as $t \in [0, T]$ and $x \in [-1, 1]$ with initial conditions $u(0, x) = -\sin(\pi x)$ and periodic boundary conditions.

## Overview

BurgersJoint is a Julia-based project that compares Fourier Galerkin, Fourier pseudospectral, Chebyshev Tau, finite difference, and neural network methods for solving Burgers’ equation.

## Modules

- `BurgersEquation`: Defines the PDE, initial conditions, and analytical solutions.
- `FourierSpectral`: Implements the Fourier Galerkin solver.
- `ChebyshevTau`: Implements the Chebyshev Tau solver.
- `FiniteDifference`: Implements the finite difference solver.
- `NeuralNetwork`: Implements the Physics-Informed Neural Network (PINN) solver.

## Examples

```julia
using BurgersJoint
x = range(-1, 1, length=100)
t = 0.5
u_analytical = BurgersEquation.analytical_solution.(x, t)
```