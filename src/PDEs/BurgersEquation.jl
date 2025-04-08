module BurgersEquation

# This module defines the PDE parameters and initial condition for Burgers' equation.
# Based on Basdevant et al. (1986), "Spectral and Finite Difference Solutions of the Burgers Equation"

using FastGaussQuadrature: gausshermite

export burgers_ic, viscosity, analytical_solution, analytical_slope, setup_time_grid

"""
    burgers_ic(x)

Default initial condition for Burgers’ equation, matching the setup in Basdevant1986:

    ``u(x, 0) = -\\sin(\\pi x)``.
"""
function burgers_ic(x)
    return -sinpi.(x)
end

"""
    viscosity()

Returns the kinematic viscosity used in Basdevant1986: ``\\nu = 1/(100\\pi)``.
"""
function viscosity()
    #return 1e-2 / pi
    return 0.003183098861837907 # Precomputed 1e-2 / pi to 16 digits
end

"""
    f(y; nu=viscosity())

Define the function ``f(y)`` as given in Basdevant1986.
"""
function f(y; nu=viscosity())
    return exp(-cos(pi * y)/ (2 * pi * nu))
end

"""
    analytical_solution(x, t; nu=viscosity(), n=50)

Computes the analytical solution ``u(x,t)``` of Burgers' equation using the
convolution form from Cole's transformation and Hermite integration. This assumes
initial condition u(x,0) = -sin(pi*x).

The solution is expressed as

``u(x,t) = \\frac{I_1(x,t)}{I_2(x,t)}``,

where

``I_1(x,t) = -\\int_{-\\infty}^{+\\infty} \\sin(\\pi(x-\\eta))\\, f(x-\\eta) \\exp\\left(-\\frac{\\eta^2}{4 \\nu t}\\right)\\, \\mathrm{d}\\eta``,

``I_2(x,t) = \\int_{-\\infty}^{+\\infty} f(x-\\eta) \\exp\\left(-\\frac{\\eta^2}{4 \\nu t}\\right)\\, \\mathrm{d}\\eta``.

Here, ``f(y) = \\exp(-\\cos(\\pi y)/(2\\pi\\nu))``. A change of variables ``\\eta = S    z`` with ``S = \\sqrt{4 \\nu t}`` is used so that

``d\\eta = S \\,\\mathrm{d}z, \\quad \\exp\\left(-\\frac{\\eta^2}{4\\nu t}\\right) = \\exp(-z^2)``.

Hermite quadrature (with ``n`` nodes, default ``n = 50``) is used to evaluate these integrals.
"""
function analytical_solution(x, t; nu=viscosity(), n=50, ic=burgers_ic)
    if t == 0
        # Return the initial condition if t=0
        return ic(x)
    end
    nodes, weights = gausshermite(n)
    S = sqrt(4 * nu * t)

    x_col = reshape(x, :, 1)
    nodes_row = reshape(nodes, 1, :) # z_i for gauss-hermite
    weights_row = reshape(weights, 1, :) # w_i for gauss-hermite

    F = f.(x_col .- nodes_row .* S; nu=nu)

    I1 = -S * sum(weights_row .* sin.(pi .* (x_col .- nodes_row .* S)) .* F, dims=2)
    I2 = S * sum(weights_row .* F, dims=2)

    return vec(I1 ./ I2)
end

"""
    analytical_slope(x, t; nu=viscosity(), n=50)

Computes the analytical slope from the analytical solution using central differences.
"""
function analytical_slope(x, t; nu=viscosity(), n=50)
    # Compute analytical solution for twice as finely spaced x
    x2 = range(-1, 1, length=2*length(x)+1)[1:end-1]
    u = analytical_solution(x2, t; nu=nu, n=n)
    dx = x2[2] - x2[1]
    # Only return the slope at the original x points
    return (u[3:2:end] - u[1:2:end-2]) / (2 * dx)
end

# Function for getting time grid and snapshot times, etc.
# It should ideally be in a module for general solver methods, but it's here for now
function setup_time_grid(dt, T, dt_snapshot = -1)
    times = collect(0.0:dt:T) # evenly spaced time steps
    to_snapshot = dt_snapshot > 0 # Flag for storing snapshots
    snapshot_time_steps = [] # Initialize empty list for snapshot time steps
    snapshot_times = []
    if to_snapshot
        # Generate list of time steps to store snapshots
        snapshot_time_steps = [round(t/dt) for t in 0:dt_snapshot:T]
        snapshot_times = [dt*i for i in snapshot_time_steps]
    end
    return times, to_snapshot, snapshot_time_steps, snapshot_times
end

end # module BurgersEquation