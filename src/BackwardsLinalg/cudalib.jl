using .CUDA
using .CUDA: CUSOLVER, CUBLAS, CURAND, abs, angle

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
    function kernel(ctx, arr)
        i, j = CUDA.@cartesianidx arr
        if i == j
            @inbounds arr[i,i] = real(arr[i,i])
        elseif j > i
            @inbounds arr[i,j] = conj(arr[j,i])
        end
        return nothing
    end

    CUDA.gpu_call(kernel, A)
    return A
end
