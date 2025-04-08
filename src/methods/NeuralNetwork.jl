module NeuralNetwork

using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, LineSearches
using ModelingToolkit: Interval
using Plots
using ProgressMeter
using ..BurgersEquation: burgers_ic, viscosity

export trainedModel, trainNeuralNetwork, solveNeuralNetwork

global trainedModel = nothing # Global variable to store the trained model

"""
    trainNeuralNetwork(N::Int; dt=1e-2, T=1.0, nu=viscosity(), ic=burgers_ic)

Train a PINN for the 1D Burgers equation on t in [0, T], x in [-1, 1].
Discretizes the domain with a time-step dt and a spatial-step dx = 2/(N-1).
Stores the trained model in the global variable `trainedModel`.
"""
function trainNeuralNetwork(N::Int; dt=1e-2, T=1.0, nu=viscosity(), ic=burgers_ic)
    # Define PDE parameters, variables, and differentials
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    # Define 1D Burgers' equation:
    #    du/dt       + u * du/dx           - nu * d^2u/dx^2  = 0
    eq = Dt(u(t, x)) + u(t, x)*Dx(u(t, x)) - nu*Dxx(u(t, x)) ~ 0
    # Note: Pretty straightforward, no?

    # Define IC and BCs
    bcs = [
        u(0, x) ~ ic(x), # Initial condition
        u(t, -1) ~ 0.0, # Left Dirichlet
        u(t, 1)  ~ 0.0, # Right Dirichlet
        u(t, -1) ~ u(t, 1) # Ensure periodic (should not be strictly necessary, but helps convergence)
    ]
    # Note: Again, very clear definition of system - love it!

    # Define temporal and spatial domains
    domains = [
        t in Interval(0.0, T),
        x in Interval(-1.0, 1.0)
    ]

    # Define neural network architecture
    chain = Chain(
        Dense(2, 16, σ),
        Dense(16, 16, σ),
        Dense(16, 1)
    )
    # Note: I initially wanted to use a Kolmogorov-Arnold network and compare to MLP.
    #       I think this should be possible by using KDense in the KolmogorovArnold.jl package.
    #       I didn't have time to play with it, however.

    # Set up training on equispaced grid with dt and dx = 2/(N-1).
    dx = 2/(N-1)
    #println("dx = $dx", " dt = $dt") # Sanity check
    strategy = GridTraining([dt,dx])
    #strategy = GridTraining(dt) # Using same spacing for both t and x trains faster, dunno why
    # Note: For real applications, you might want to use a more complex strategy
    #       e.g. stochastic, quasi-random, or quadrature sampling.

    # Define complete PDE system
    indvars = [t, x]
    depvars = [u(t, x)]
    @named pde_system = PDESystem(eq, bcs, domains, indvars, depvars)

    # Set up PINN for solving the PDE system
    discretization = PhysicsInformedNN(chain, strategy)
    sym_prob = symbolic_discretize(pde_system, discretization)

    # Define loss functions
    pde_loss_functions = sym_prob.loss_functions.pde_loss_functions
    bc_loss_functions  = sym_prob.loss_functions.bc_loss_functions
    loss_functions     = [pde_loss_functions; bc_loss_functions]
    loss_function(θ, p) = sum(map(l -> l(θ), loss_functions))

    # Set number of iterations to train for
    maxiters = 3000

    # Define callback with progress info and storage of loss history
    # Each iteration in the solver calls the callback
    tot_loss_history = Float64[]
    pde_loss_history = Float64[]
    bc_loss_history  = Float64[]
    progress_bar = Progress(maxiters, desc="Training PINN") # Progress bar for training
    #iter = 0 # Uncomment for printout
    function callback(p, l)
        next!(progress_bar)   # Advance progress bar
        #iter += 1 # Uncomment for printout

        # Get and store loss and its components
        curr_tot_loss = first(l)
        curr_pde_loss = first(map(l_ -> l_(p.u), pde_loss_functions))
        curr_bc_loss  = first(map(l_ -> l_(p.u), bc_loss_functions))

        push!(tot_loss_history, curr_tot_loss)
        push!(pde_loss_history, curr_pde_loss)
        push!(bc_loss_history, curr_bc_loss)

        # Optional logging printout (requires uncommenting the lines with iter as well)
        #@info "Iteration: $(iter)/$(maxiters) total loss: $(first)  pde loss: $curr_pde_loss  bc loss: $curr_bc_loss"

        return false
    end

    # Create the optimization problem
    f_ = OptimizationFunction(loss_function, AutoZygote())
    prob = OptimizationProblem(f_, sym_prob.flat_init_params)

    # Train the model using BFGS with backtracking line search
    res = solve(prob, BFGS(linesearch=BackTracking()); maxiters=maxiters, callback=callback)

    # Finish the progress bar
    finish!(progress_bar)

    # Plot convergence of losses
    iters = 1:length(tot_loss_history)
    plt_losses = plot(
        iters, tot_loss_history,
        label = "Total loss",
        title = "Loss convergence",
        xlabel = "Iteration",
        ylabel = "Loss",
        yscale = :log10
    )
    plot!(plt_losses, iters, pde_loss_history, label = "PDE loss")
    plot!(plt_losses, iters, bc_loss_history, label = "BC loss")
    display(plt_losses)

    # Store the trained model result so solveNeuralNetwork can use it
    global trainedModel = (res = res, phi = sym_prob.phi)

    return trainedModel
end


"""
    solveNeuralNetwork(N::Int; dt=1e-2, T=1.0, nu=viscosity(), ic=burgers_ic)

Solves the 1D Burgers equation on t in [0, T], x in [-1, 1] using a physics-informed neural network (PINN).
If no trained model exists, it will call `trainNeuralNetwork(N; dt=dt, T=T, nu=nu, ic=ic)` first.
The predicted solution is evaluated at a grid of N points in x and at times t = 0, 1/pi, 2/pi, ..., T.

Arguments:
- N: Number of grid points in the spatial domain.

Keyword arguments (optional):
- dt: Time step for the training (default = 1e-2).
- T: Duration to solve over (default = 1.0).
- nu: Kinematic viscosity, only used for training (default = BurgersEquation.viscosity()).
- ic: Initial condition, only used for training (default = BurgersEquation.burgers_ic()).

Output:
- x: Spatial grid of length N.
- u_final: Final solution in real space (length N).
- snapshot_times: Vector of time steps at which the solution was evaluated.
- history: Vector of solution snapshots at times t = n/pi, n=0,1,2,...
"""
function solveNeuralNetwork(N::Int; dt=1e-2, T=1.0, nu=viscosity(), ic=burgers_ic)
    # If not trained, train it now
    if trainedModel === nothing
        @info "No trained model found. Training now."
        trainNeuralNetwork(N; dt=dt, T=T, nu=nu, ic=ic)
    else
        @info "Using existing trained model."
    end

    # Unpack trained model
    res = trainedModel[:res]
    phi = trainedModel[:phi]

    x = range(-1, 1, length=N) # Spatial grid of length N

    # Generate list of times to evaluate the model at
    snapshot_times = 0:1/pi:T
    if snapshot_times[end] < T
        snapshot_times = vcat(snapshot_times, T)
    end

    # Evaluate the model at each time and store the results
    u_history = [ [ first(phi([t, xx], res.u)) for xx in x ] for t in snapshot_times ]
    u_final = u_history[end]

    return x, u_final, snapshot_times, u_history
end

end  # module NeuralNetwork

