import Base: *, +, -, ctranspose, zero 
*(J::Base.LinAlg.UniformScaling, A::FixedSizeArrays.FixedArray) = J.λ*A
*(A::FixedSizeArrays.FixedArray, J::Base.LinAlg.UniformScaling) = A*J.λ
+{m, n, T}(A::Mat{m,n, T}, J::Base.LinAlg.UniformScaling) = A + J.λ*eye(Mat{m,n,T})
+{m, n, T}(J::Base.LinAlg.UniformScaling, A::Mat{m,n, T}) = J.λ*eye(Mat{m,n,T}) + A
-{m, n, T}(A::Mat{m,n, T}, J::Base.LinAlg.UniformScaling) = A - J.λ*eye(Mat{m,n,T})
-{m, n, T}(J::Base.LinAlg.UniformScaling, A::Mat{m,n, T}) = J.λ*eye(Mat{m,n,T}) - A

zero{T, NDim, SIZE}(_::FixedSizeArrays.FixedArray{T,NDim,SIZE}) = zero(typeof(_))
#dot{N,T}(a::T, b::FixedSizeArrays.Vec{N,T}) = a'*b

#=
@generated function *{T, T2, R, C}(a::Mat{R, C, T}, b::Vec{C,T2})
   expr = [:(dot(row(a, $i), b.(1))) for i=1:R]
   return quote 
       $(Expr(:boundscheck, false))
       Vec($(expr...))
   end
end
@generated function *{T, M, N, R}(a::Mat{M, N, T}, b::Mat{N, R, T})
   expr = Expr(:tuple, [Expr(:tuple, [:(dot(row(a, $i), column(b,$j))) for i in 1:M]...) for j in 1:R]...)
   return quote 
       $(Expr(:boundscheck, false))
       Mat($(expr))
   end
end
=#

ctranspose(v::Vec) =  Mat((v._,))'

function cumsum0(dx::Vector)
        n = length(dx) + 1
        x = similar(dx, n)
        x[1] = 0.0      
        for i in 2:n
                x[i] = x[i-1] + dx[i-1] 
        end
        x
end

import Base.Random: randn

randn(::Type{Float64}) = randn()
randn{T}(::Type{Complex{T}}) = Complex(randn(T), randn(T))
immutable RandnFunctor{T} <: FixedSizeArrays.Func{1} end
@inline call{T}(rf::Type{RandnFunctor{T}}, i...) = randn(T)
@inline randn{FSA <: FixedArray}(x::Type{FSA}) = map(RandnFunctor{eltype(FSA)}, FSA)


