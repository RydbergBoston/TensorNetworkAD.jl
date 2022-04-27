using CUDA
using CUDA: CUSOLVER, CUBLAS, CURAND, abs, angle

match_type(m::CuArray, ::CuMatrix) = m
do_adjoint(A::CuMatrix) = CuMatrix(A')
trtrs!(c1::Char, c2::Char, c3::Char, r::CuArray, b::CuVector) = CUBLAS.trsv!(c1, c2, c3, r, b)
trtrs!(c1::Char, c2::Char, c3::Char, r::CuArray{T}, b::CuMatrix{T}) where T = CUBLAS.trsm!('L',c1, c2, c3, T(1), r, b)

function symeigen(a::CuArray{<:Complex})
    CUSOLVER.heevd!('V','U',copy(a))
end

function symeigen(a::CuArray{<:Real})
    CUSOLVER.syevd!('V','U',copy(a))
end

function qr(a::CuArray)
    M, N = size(a)
    A, tau = CUSOLVER.geqrf!(copy(a))
    if M > N
        R = CUDA.triu(A[1:N,:])
    else
        R = CUDA.triu(A)
    end
    Q = CUSOLVER.orgqr!(A, tau)
    return Q, R
end

"""
    copyltu!(A::AbstractMatrix) -> AbstractMatrix

copy the lower triangular to upper triangular.
"""
function copyltu!(A::CuArray)
    CUDA.gpu_call(A; name="copyltu!") do ctx, A
        idx = CUDA.@cartesianidx A
        @inbounds if idx[1] == idx[2]
            A[idx] = real(A[idx])
        elseif idx[2] > idx[1]
            A[idx] = conj(A[idx[2], idx[1]])
        end
        return nothing
    end
    return A
end
