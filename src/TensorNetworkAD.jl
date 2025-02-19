module TensorNetworkAD
using Zygote
using OMEinsum

export trg, num_grad
export ctmrg
export optimiseipeps
export hamiltonian, model_tensor, mag_tensor
export Ising, TFIsing, Heisenberg
export BackwardsLinalg

include("BackwardsLinalg/BackwardsLinalg.jl")

include("hamiltonianmodels.jl")

include("trg.jl")
include("fixedpoint.jl")
include("ctmrg.jl")
include("ipeps.jl")
include("autodiff.jl")
include("variationalipeps.jl")
include("exampletensors.jl")

end # module
