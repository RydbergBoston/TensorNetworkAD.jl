using Test
using CUDA

@testset "qr" begin
    include("qr.jl")
end

@testset "svd" begin
    include("svd.jl")
end

@testset "eigen" begin
    include("eigen.jl")
end

if CUDA.functional()
    @testset "eigen" begin
        include("cudalib.jl")
    end
end