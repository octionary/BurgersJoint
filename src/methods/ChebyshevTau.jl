module ChebyshevTau

using LinearAlgebra
using ProgressMeter
using ..BurgersEquation: burgers_ic, viscosity

export solveChebyshevTau

# Helper function to compute the Chebyshev differentiation matrix
function chebyshev_diff_matrix(N, x)
    # I base use equation (67) from the note on spectral methods
    # x is the vector of Chebyshev nodes of length N+1.
    D = zeros(Float64, N+1, N+1)
    c = ones(Float64, N+1) # Weight factors for differentiation: c_0 = c_N = 2, c_n = 1 for 0 < n < N
    c[1] = 2.0; c[end] = 2.0
    # First evaluate for i != j
    for i in 1:(N+1)
        for j in 1:(N+1)
            if i != j
                D[i, j] = (c[i] / c[j]) * ((-1)^(i+j)) / (x[i] - x[j])
            end
        end
    end
    # Then for 1 <= i = j <= N
    for i in 2:N
        D[i, i] = -x[i] / (2*(1 - x[i]^2))
    end
    # Lastly for i = j = 0 and i = j = N
    D[1, 1] = (2N^2 + 1) / 6
    D[N+1, N+1] = -D[1, 1]
    return D
end

# Helper function to find operator for viscous term
function linear_operator(N, nu)
    Lmat = zeros(N-1, N+1) # each row corresponds to a given n = 0,1,...,N-2
    for n in 0:(N-2)
        row = n + 1
        for m in (n+2):N
            # Include only contributions where m+n is even
            if iseven(m + n)
                Lmat[row, m+1] = nu * m * (m^2 - n^2)
            end
        end
    end
    return Lmat
end

# Helper function to compute the nonlinear term
function nonlinear_term!(N_coeffs,a, V, D)
    u = V * a # Compute u from the spectral coefficients a
    du_dx = D * u # Compute du/dx from u
    f = u .* du_dx # Viscous term 
    N_coeffs .= V \ f # Transform back to spectral coefficients
    return N_coeffs
end

"""
    solveChebyshevTau(N; dt=1e-3, T=1.0, nu=viscosity(), ic=burgers_ic)

Solves the 1D Burgers equation on t in [0, T], x in [-1, 1] using the Chebyshev-Tau spectral method 
with the ABCN scheme: Adams-Bashforth for the advective term and Crank-Nicholson for the viscous term,
as described in Basdevant et al. (1986). Chebyshev-Gauss-Lobatto collocation spatial grid is used.

Arguments:
- N: Number of grid points (also number of Chebyshev modes).

Keyword arguments:
- dt: Time step.
- T: Duration to solve over.
- dt_snapshot: Time step for storing snapshots (default = T, only initial and final).
- nu: Kinematic viscosity (defaults to BurgersEquation.viscosity()).
- ic: Initial condition (defaults to BurgersEquation.burgers_ic()).

Output:
- x: Real-space grid of length N.
- u_final: Final solution in real space (length N).
- times: Vector of time steps.
- history: Vector of solution snapshots at times t = n/pi, n=0,1,2,...
"""
function solveChebyshevTau(N; dt = 1e-3, T = 1.0, dt_snapshot = T, nu = viscosity(), ic = burgers_ic)
    # Generate Chebyshev-Gauss-Lobatto collocation points: x_j = cos(pi * j / N), j=0,1,...,N
    x = cos.(pi * (0:N) / N) 

    # Compute initial coefficients from the initial condition
    u = ic.(x) # get initial u
    V = cos.((0:N)' .* acos.(x)) # Vandermonde matrix for Chebyshev polynomials, so that u = V * a
    a = V \ u # Solve for coefficients a

    # Set up time-stepping
    times = collect(0.0:dt:T)
    history = Vector{Vector{Float64}}() # store snapshots of the solution
    push!(history, copy(u)) # store initial condition

    # Define weight factors: c_0 = 2, c_n = 1 for n>= 1
    c = ones(N+1) # + 1 for the zeroth term
    c[1] = 2

    # Compute the Chebyshev differentiation matrix, D_ij = T'_j(x_i) for i,j=0,1,...,N
    D = chebyshev_diff_matrix(N, x)

    # Assemble the linear operator Lmat such that L = Lmat * a
    Lmat = linear_operator(N, nu)

    # Compute initial nonlinear term: spectral coefficients of -u*du/dx
    N_curr = zeros(N+1)
    nonlinear_term!(N_curr, a, V, D)
    N_prev = copy(N_curr) # bootstrap the first step

    # Construct total matrix for the implicit part of the time-stepping: A = c / dt * I - Lmat / 2
    # Note to self: Remember to explain this form in the notebook
    A = zeros(N+1, N+1)
    A[1:N-1, 1:N-1] = Diagonal(c[1:N-1] ./ dt)
    A[1:N-1, 1:N+1] -= 0.5 * Lmat

    # Use two last rows to enforce BCs
    A[N, :] .= 1 # Enforce sum(a) = 0
    for m in 0:N
        # Enforce sum((-1)^n * a_n) = 0
        A[N+1, m+1] = (-1)^m
    end

    # Factorize A (speeds up solving the linear system)
    A_fact = factorize(A)

    # Time-stepping loop
    t_step_last_stored = 0 # Last time step at which a snapshot was stored
    N_AB = similar(N_curr) # Adams-Bashforth term
    L = zeros(N-1) # Linear term
    c_over_dt = c[1:N-1] ./ dt
    b = zeros(N+1) # RHS of the linear system
    p = Progress(length(times)-1, dt=0.5, desc="Solving: ", barglyphs=BarGlyphs("[=> ]")) # Progress bar
    @inbounds for t_step in 2:length(times)
        N_AB .= 1.5 * N_curr - 0.5 * N_prev # Compute the Adams-Bashforth term: 3/2 * N_curr - 1/2 * N_prev

        mul!(L, Lmat, a) # Compute the linear term L = Lmat * a

        # Construct the RHS of the linear system
        for i in 1:N-1
            b[i] = c_over_dt[i] * a[i] + 0.5 * L[i] - N_AB[i]
            # Loop should be faster than vectorized here, avoiding temporary arrays
        end
        b[N:N+1] .= 0 # Enforce BCs

        # Solve the linear system
        a .= A_fact \ b

        # Compute the nonlinear term
        N_prev .= N_curr
        nonlinear_term!(N_curr, a, V, D)

        # If time is a multiple of 1/pi, store the solution
        mod_time = abs(mod(times[t_step], 1/pi))
        if (mod_time < dt/2) && (t_step > t_step_last_stored)
            t_step_last_stored = t_step
            u .= V * a
            push!(history, copy(u))
        end

        next!(p) # Update the progress bar
    end

    # Get final solution
    if t_step_last_stored < length(times)
        u .= V * a
        push!(history, copy(u))
    end

    return x, u, times, history
end # function solveChebyshevTau


end # module ChebyshevTau
