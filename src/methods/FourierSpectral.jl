module FourierSpectral

using FFTW
using ProgressMeter
using ..BurgersEquation: burgers_ic, viscosity

export solveFourierSpectral

"""
    solveFourierSpectral(N; dt=1e-3, T=1.0, nu=viscosity(), ic=burgers_ic, dealias_factor=2/3)

Solves the 1D Burgers equation on t in [0, T], x in [-1, 1] using Fourier methods (Galerkin or 
pseudospectral). It is based on the method described in Basdevant et al. (1986) and uses their
leapfrog-like scheme for time integration. 

Arguments:
- N: Number of grid points (also number of Fourier modes before dealiasing).

Keyword arguments:
- dt: Time step.
- T: Duration to solve over.
- dt_snapshot: Time step for storing snapshots (default = -1, only initial state).
- nu: Kinematic viscosity (defaults to BurgersEquation.viscosity()).
- ic: Initial condition (defaults to BurgersEquation.burgers_ic()).
- dealias_factor: Factor for dealiasing
    - 2/3 corresponsd to 2/3-rule Fourier Galerkin (default, aliasing errors removed).
    - 1 corresponds to Fourier pseudospectral (aliasing errors will be present).

Output:
- x: Real-space grid of length N.
- u_final: Final solution in real space (length N).
- snapshot_times: Vector of time steps at which the solution was saved.
- snapshots: Vector of solution snapshots at times t = {j*dt_snapshot}, j=0,1,2,...
"""
function solveFourierSpectral(N::Int; dt=1e-3, T=1.0, dt_snapshot = -1, nu=viscosity(), ic=burgers_ic, dealias_factor=2/3)
    # Note: 
    #   Unlike in Basdevant1986, I use N as the total number of grid points instead of the number of 
    #   positive/negative grid points after dealiasing. I use dt as the full step time, whereas 
    #   Basdevant uses the half-step time Delta t = dt/2.

    # Domain setup
    L = 2.0 # Length of the domain
    x = range(-1, 1, length=N+1)[2:end]  # N points, including right, not left endpoint 
    # Note:
    #   The above grid avoids including the periodic endpoint twice which 
    #   would effectively have expanded the grid by one step to the right of x=1.

    # Angular wavenumber array for the FFT (FFTW uses wrap-around order)
    j_list = [] # List of factors for the angular wavenumber
    if N % 2 == 0
        j_list = [0:floor(Int, N/2); -floor(Int, N/2)+1:-1]
    else
        j_list = [0:floor(Int, (N-1)/2); -floor(Int, (N-1)/2):-1]
    end
    k = 2 * pi / L * j_list # Angular wavenumber array
    j_max = maximum(j_list) # Maximum wavenumber factor (N/2 or (N-1)/2 for even/odd N)
    # Note:
    #   The 2pi/L spacing between wavenumbers is due to FFTW convention.
    #   Basdevant1986 uses integer spacing.

    # Initial condition in real space and Fourier space
    u_real = [ic(xj) for xj in x]
    u_hat = rfft(u_real)

    # Dealiasing: zero out large wave numbers
    j_cut = floor(Int, dealias_factor * j_max) # index of the cut-off frequency
    k_cut = j_cut * 2*pi / L
    function dealias!(uhat)
        for j in eachindex(uhat)
            if abs(k[j]) > k_cut
                uhat[j] = 0
            end
        end
    end
    dealias!(u_hat)
    # Note:
    #   j_cut is called N in Basdevant1986 and is the factor of the cut-off frequency, k_cut. 
    #   j_max is the maximum wavenumber factor, corresponding to M in Basdevant1986. 
    #   j_cut is found by:
    #   j_cut = (j_cut/M) * M = dealias_factor * j_max, where dealias_factor = (j_cut/M) by definition.
    #   So with the 2/3 rule, we have j_cut = (2/3) * j_max

    # Setup time integration
    times = collect(0.0:dt:T) # evenly spaced time steps
    snapshots = Vector{Vector{Float64}}()
    push!(snapshots, copy(u_real))  # Store initial condition as first snapshot
    to_snapshot = dt_snapshot > 0 # Flag for storing snapshots
    snapshot_time_steps = [] # Initialize empty list for snapshot time steps
    snapshot_times = []
    if to_snapshot
        # Generate list of time steps to store snapshots
        snapshot_time_steps = [round(t/dt) for t in 0:dt_snapshot:T]
        snapshot_times = [dt*i for i in snapshot_time_steps]
    end

    # Precompute exponential factors
    lambda = -nu /pi * k.^2
    expfac = [exp(lambda[j]*dt) for j in 1:length(u_hat)] # exp(2 * lambda * Delta t) in Basdevant1986
    expfac_half = [exp(lambda[j]*dt/2) for j in 1:length(u_hat)] # exp(lambda * Delta t) in Basdevant1986
    # Note:
    #   dt is full-step time, not half-step, explaining factor 2 difference to Basdevant1986.

    # Nonlinear term function, script N_k in Basdevant1986
    function nonlinear_term(u_hat_curr)
        u_real .= irfft(u_hat_curr, length(u_real)) # Backward FFT to real space
        @inbounds for i in eachindex(u_real)
            u_real[i] = u_real[i]^2  # u^2 calculated in real space
        end
        w_hat = rfft(u_real) # Forward FFT to get w Fourier space
        NL_hat = similar(u_hat_curr) # Nonlinear term in Fourier space (non-script N_k)
        for j in eachindex(w_hat)
            NL_hat[j] = -0.5 * (im * k[j]) * w_hat[j] # scriptN_k= -ik/2 * N_k in Basdevant1986
        end
        dealias!(NL_hat) # Dealiasing
        return NL_hat
    end

    NL_hat = nonlinear_term(u_hat) # Initial nonlinear term

    # First step: explicit forward Euler
    #   We have u_hat(0) and NL_hat(0), but to kickstart we also need u_hat(dt/2) and NL_hat(dt/2).
    #   I use a forward Euler step, while Basdevant1986 does not explicitly state how they get it:
    #       u_k(dt/2) = exp(lambda_k*dt/2) * (u_k(0) + dt * NL_k(0))
    u_hat_half = similar(u_hat)
    @inbounds for j in eachindex(u_hat)
        u_hat_half[j] = expfac_half[j] * (u_hat[j] + dt * NL_hat[j])
    end
    NL_hat_half = nonlinear_term(u_hat_half)

    # Time integration using a "leapfrog"-like scheme
    #p = Progress(length(times))
    @inbounds for n in eachindex(times)
        # First finish full step
        for j in eachindex(u_hat)
            u_hat[j] = expfac[j]*u_hat[j] + dt*expfac_half[j]*NL_hat_half[j]
        end
        NL_hat .= nonlinear_term(u_hat)

        # Store solution snapshot if on a snapshot time step
        if to_snapshot 
            if n in snapshot_time_steps
                u_real .= irfft(u_hat, length(u_real))
                push!(snapshots, copy(u_real))
            end
        end
        
        # Then do half of next step
        for j in eachindex(u_hat_half)
            u_hat_half[j] = expfac[j]*u_hat_half[j] + dt*expfac_half[j]*NL_hat[j]
        end
        NL_hat_half .= nonlinear_term(u_hat_half)

        #next!(p) # Update the progress bar
    end
    # Note:
    #   I'm not sure this counts as an actual leapfrog scheme, as both u_hat and NL_hat
    #   is evaluated and full steps as well as half-steps. It seems this is what Basdevant
    #   does and calls leapfrog, though they don't state explicity how they get NL_hat at
    #   half-steps. Maybe N_hat(t+dt/2) could be approximated by N_hat(t), cutting the number
    #   of evaluations of the nonlinear term in half.

    # Final inverse transform to real space
    u_real .= irfft(u_hat, length(u_real))

    return x, u_real, snapshot_times, snapshots
end # function solveFourierSpectral

end # module FourierSpectral
