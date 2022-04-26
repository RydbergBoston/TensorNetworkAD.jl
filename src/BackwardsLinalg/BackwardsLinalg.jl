module BackwardsLinalg
import LinearAlgebra
using Requires

struct ZeroAdder end
Base.:+(a, zero::ZeroAdder) = a
Base.:+(zero::ZeroAdder, a) = a
Base.:-(a, zero::ZeroAdder) = a
Base.:-(zero::ZeroAdder, a) = -a
Base.:-(zero::ZeroAdder) = zero

include("qr.jl")
include("svd.jl")
include("rsvd.jl")
include("symeigen.jl")
include("zygote.jl")

function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" include("cudalib.jl")
end
end
