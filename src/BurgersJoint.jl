module BurgersJoint

include("PDEs/BurgersEquation.jl")
include("methods/FourierSpectral.jl")
include("methods/ChebyshevTau.jl")
include("methods/FiniteDifference.jl")
include("methods/NeuralNetwork.jl")

export BurgersEquation, FourierSpectral, ChebyshevTau, FiniteDifference, NeuralNetwork

end # module BurgersJoint