module FourierSpectral

using FFTW
using ..BurgersEquation: burgers_ic, viscosity

export solveFourier

"""
    solveFourier(Nx; dt=1e-3, tmax=2.0, nu=viscosity(), ic=burgers_ic, alias_factor=2/3)

Solve the 1D Burgers equation on ``x \\in [-1,1]`` using a Fourier-Galerkin method 
(based on Basdevant1986).

Arguments:
- Nx: Number of grid points (also number of Fourier modes).

Keyword arguments:
- dt: Time step.
- tmax: Final simulation time.
- nu: Kinematic viscosity (defaults to BurgersEquation.viscosity()).
- alias_factor: Factor for dealiasing (default = 2/3 for the 2/3 rule Fourier Galerkin).
                Use 1 for Fourier pseudospectral.

Output:
- x: Real-space grid of length Nx.
- u_final: Final solution in real space (length Nx).
- times: Vector of time steps.
- history: Vector of solution snapshots at times t = n/pi, n=0,1,2,...
"""
function solveFourier(Nx; dt=1e-3, tmax=2.0, nu=viscosity(), ic=burgers_ic, alias_factor=2/3)
    # Note: 
    #   I use Nx (the number of grid points) instead of M=Nx/2 like Basdevant1986.
    #   I use dt as the full step time, whereas Basdevant uses the half-step time Delta t = dt/2

    # Domain setup
    Lx = 2.0
    x = range(-1, 1, length=Nx+1)[1:end-1]  # Nx points

    # Frequency array for the FFT (FFTW uses wrap-around order)
    k = [0:floor(Nx/2); -floor(Nx/2)+1:-1] * 2*pi / Lx
    # Note:
    #   The 2pi/Lx factor matches the FFTW convention, whereas Basdevant1986 uses integer k.

    # Initial condition in real space and Fourier space
    u_real = [ic(xj) for xj in x]
    u_hat = rfft(u_real)

    # Dealiasing: zero out large wave numbers
    N_cut = floor(Int, Nx * alias_factor / 2) # index of the cut-off frequency
    k_cut = N_cut * 2*pi / Lx
    function dealias!(uhat)
        for j in eachindex(uhat)
            if abs(k[j]) > k_cut
                uhat[j] = 0
            end
        end
    end
    dealias!(u_hat)
    # Note:
    #   The index of the cut-off frequency N_cut (or N in Basdevant1986) is then 
    #   N_cut = M * (N/M) = Nx * (N/M)/2 = Nx * alias_factor/2, alias_factor = (N/M).
    #   So with the 2/3 rule, we have alias_factor = 2/3.

    # Setup time integration
    tsteps = collect(0.0:dt:tmax)
    history = Vector{Vector{Float64}}()
    push!(history, copy(u_real))  # Store initial condition as first snapshot

    # Precompute exponential factors
    lambda = -nu /pi * k.^2
    expfac = [exp(lambda[j]*dt) for j in 1:length(u_hat)] # exp(2 * lambda * Delta t) in Basdevant1986
    expfac_half = [exp(lambda[j]*dt/2) for j in 1:length(u_hat)] # exp(lambda * Delta t) in Basdevant1986
    # Note:
    #   dt is full-step time, not half-step, explaining factor 2 difference to Basdevant1986.

    # Temporary buffer for real-space computations
    buffer_real = similar(u_real)

    # Nonlinear term function, script N_k in Basdevant1986
    function nonlinear_term(u_hat_now)
        buffer_real .= irfft(u_hat_now, length(buffer_real)) # backward FFT to real space
        @inbounds for i in eachindex(buffer_real)
            buffer_real[i] = buffer_real[i]^2  # u^2
        end
        w_hat = rfft(buffer_real) # forward FFT to get w Fourier space
        NL_hat = similar(u_hat_now) # nonlinear term in Fourier space (non-script N_k)
        for j in eachindex(w_hat)
            NL_hat[j] = -0.5 * (im * k[j]) * w_hat[j] # scriptN_k= -ik/2 * N_k in Basdevant1986
        end
        dealias!(NL_hat) # dealiasing
        return NL_hat
    end

    NL_hat = nonlinear_term(u_hat) # Initial nonlinear term

    # First step: explicit forward Euler
    #   We have u_hat(0) and NL_hat(0), but to kickstart we also need u_hat(dt/2) and NL_hat(dt/2).
    #   I choose to use a forward Euler step:
    #       u_k(dt/2) = exp(lambda_k*dt/2) * (u_k(0) + dt * NL_k(0))
    u_hat_half = similar(u_hat)
    for j in eachindex(u_hat)
        u_hat_half[j] = expfac_half[j] * (u_hat[j] + dt * NL_hat[j])
    end
    NL_hat_half = nonlinear_term(u_hat_half)

    # Time integration using a "leapfrog"-like scheme
    for n in Iterators.drop(eachindex(tsteps), 1)

        # First do full step
        for j in eachindex(u_hat)
            u_hat[j] = expfac[j]*u_hat[j] + dt*expfac_half[j]*NL_hat_half[j]
        end
        NL_hat = nonlinear_term(u_hat)
        
        # Then do half step
        for j in eachindex(u_hat_half)
            u_hat_half[j] = expfac[j]*u_hat_half[j] + dt*expfac_half[j]*NL_hat[j]
        end
        NL_hat_half = nonlinear_term(u_hat_half)

        # If time is a multiple of 1/pi, store the solution
        if abs(mod(tsteps[n], 1/pi)) < dt/2
            buffer_real .= irfft(u_hat, length(buffer_real))
            push!(history, copy(buffer_real))
        end
    end
    # Note:
    #   I'm not sure this counts as an actual leapfrog scheme, as both u_hat and NL_hat
    #   is evaluated and full steps as well as half-steps. It seems this is what Basdevant
    #   does and calls leapfrog, though they don't state explicity how they get NL_hat at
    #   half-steps. Maybe N_hat(t+dt/2) could be approximated by N_hat(t), cutting the number
    #   of nonlinear term evaluations in half.

    # Final inverse transform to real space
    u_real .= irfft(u_hat, length(u_real))
    push!(history, copy(u_real))

    return x, u_real, tsteps, history
end

end # module FourierSpectral
