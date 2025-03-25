module BurgersJoint

include("PDEs/BurgersEquation.jl")
include("methods/FourierSpectral.jl")

export BurgersEquation, FourierSpectral

end # module BurgersJoint