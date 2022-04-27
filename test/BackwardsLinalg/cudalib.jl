using CUDA
CUDA.allowscalar(false)  # important to prevent element-wise operation to GPU Arrays directly.
using LinearAlgebra
using TensorNetworkAD.BackwardsLinalg
using TensorNetworkAD.BackwardsLinalg: svd_back, qr_back, symeigen_back, trtrs!, mpow2
using Test, Random

@testset "cuda qr" begin
    Random.seed!(7)
    CUDA.seed!(7)
    a = randn(ComplexF64, 5, 5) |> CuArray
    @test Matrix(copyltu!(copy(a))) ≈ copyltu!(Matrix(a))
    for T in [Float32, ComplexF32, Float64, ComplexF64]
        for sz in [(2,4), (4,2), (4,4)]
            a = CuArray(randn(T, sz...))
            Q, R = BackwardsLinalg.qr(a)
            @test Matrix(Q'*Q) ≈ I
            @test Matrix(Q*R) ≈ Matrix(a)
            @test Matrix(a) ≈ Matrix(a)
            dQ, dR = CUDA.rand(T, size(Q)...), CUDA.rand(T, size(R)...)
            @test Matrix(qr_back(a, Q, R, dQ, dR)) ≈ qr_back(Matrix(a), Matrix(Q), Matrix(R), Matrix(dQ), Matrix(dR))
        end
    end
end

@testset "cuda svd" begin
    Random.seed!(5)
    CUDA.seed!(5)
    for T in [Float32, ComplexF32, Float64, ComplexF64]
        for sz in [(2,4), (4,2), (4,4)]
            a = randn(T, sz...) |> CuArray
            U, S, V = BackwardsLinalg.svd(a)
            @test Matrix(U * Diagonal(S) * V') ≈ Matrix(a)
            dU, dS, dV = CUDA.rand(T, size(U)...), CUDA.rand(T, size(S)...), CUDA.rand(T, size(V)...)
            @test Matrix(svd_back(U, S, V, dU, dS, dV)) ≈ svd_back(Matrix(U), Vector(S), Matrix(V), Matrix(dU), Vector(dS), Matrix(dV))
        end
    end
end

@testset "cuda symeigen" begin
    Random.seed!(7)
    CUDA.seed!(7)
    for T in [Float32, Float64, ComplexF32, ComplexF64]
        a = randn(T, 5,5) |> CuArray; a += a'
        E, U = BackwardsLinalg.symeigen(a)
        @test U * Diagonal(E) * U' ≈ a

        dE, dU = CUDA.rand(T, size(E)...), CUDA.rand(T, size(U)...)
        @test Matrix(symeigen_back(E, U, dE, dU; η=1e-20)) ≈ Matrix(symeigen_back(Vector(E), Matrix(U), Vector(dE), Matrix(dU)))
    end
end

@testset "trtrs!" begin
    Random.seed!(3)
    for T in [Float32, Float64, ComplexF32, ComplexF64]
        bv = randn(T, 5) |> CuArray
        bm = randn(T, 5, 5) |> CuArray
        for b in [bv, bm]
            a = randn(T, 5,5) |> CuArray
            res = trtrs!('U', 'N', 'N', copy(a), copy(b))
            @test Array(res) ≈ trtrs!('U', 'N', 'N', Matrix(a), Array(b))
            @test eltype(res) == T
        end
    end
end
