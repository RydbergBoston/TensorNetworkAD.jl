module BackwardsLinalg
import LinearAlgebra

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
include("cudalib.jl")
end
