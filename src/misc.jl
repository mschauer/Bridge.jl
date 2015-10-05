import Base: *, +, -, /, \, ctranspose, zero, dot, chol, trace
*(J::Base.LinAlg.UniformScaling, A::FixedSizeArrays.FixedArray) = J.λ*A
*(A::FixedSizeArrays.FixedArray, J::Base.LinAlg.UniformScaling) = A*J.λ
/(A::FixedSizeArrays.FixedArray, J::Base.LinAlg.UniformScaling) = A/J.λ
\(J::Base.LinAlg.UniformScaling,v::FixedSizeArrays.FixedVector) = v/J.λ

chol(J::Base.LinAlg.UniformScaling, ::Type{Val{:U}}) = sqrt(J.λ)*I
chol(J::Base.LinAlg.UniformScaling, ::Type{Val{:L}}) = sqrt(J.λ)*I
chol(z::Float64, ::Type{Val{:L}}) = sqrt(z)

+{m, n, T}(A::Mat{m,n, T}, J::Base.LinAlg.UniformScaling) = A + J.λ*eye(Mat{m,n,T})
+{m, n, T}(J::Base.LinAlg.UniformScaling, A::Mat{m,n, T}) = J.λ*eye(Mat{m,n,T}) + A
-{m, n, T}(A::Mat{m,n, T}, J::Base.LinAlg.UniformScaling) = A - J.λ*eye(Mat{m,n,T})
-{m, n, T}(J::Base.LinAlg.UniformScaling, A::Mat{m,n, T}) = J.λ*eye(Mat{m,n,T}) - A

zero{T, NDim, SIZE}(_::FixedSizeArrays.FixedArray{T,NDim,SIZE}) = zero(typeof(_))

dot(J::Base.LinAlg.UniformScaling{Float64}, b::Float64) = J.λ*b


function cumsum0(dx::Vector)
        n = length(dx) + 1
        x = similar(dx, n)
        x[1] = 0.0      
        for i in 2:n
                x[i] = x[i-1] + dx[i-1] 
        end
        x
end

import Base: randn, rand


randn(::Type{Float64}) = randn()
randn{T}(::Type{Complex{T}}) = Complex(randn(T), randn(T))
immutable RandnFunctor{T} <: FixedSizeArrays.Func{1} end
@inline call{T}(rf::Type{RandnFunctor{T}}, i...) = randn(T)
@inline randn{FSA <: FixedArray}(x::Type{FSA}) = map(RandnFunctor{eltype(FSA)}, FSA)

chol{T}(m::Mat{1,1,T},::Type{Val{:L}}) = Mat(chol(m[1,1], Val{:L}))
function chol{T}(m::Mat{2,2,T},::Type{Val{:L}})
    m[1,2]==m[2,1]' || throw(Error("Matrix not symmetric"))
    l11 = chol(m[1,1], Val{:L})
    @fsa    [l11        inv(l11)*m[1,2]
            zero(T) chol(m[2,2] - (m[2,1]*inv(m[1,1])*m[1,2]), Val{:L})]
end

function chol{T}(m::Mat{2,2,T},::Type{Val{:R}})
    m[1,2]==m[2,1]' || throw(Error("Matrix not symmetric"))
    l11 = sqrt(m[1,1])
    @fsa    [l11   zero(T) 
            m[2,1]*inv(l11) chol(m[2,2] - (m[2,1]*inv(m[1,1])*m[1,2]), Val{:L})]
end


function sumlogdiag{m,T}(A::Mat{m,m,T}, d=m) 
    t = zero(T)
    for i in 1:m
        t += log(A[i,i])
    end
    t
end    
sumlogdiag(a::Float64, d=1) = log(a)
sumlogdiag(A,d) = sum(log(diag(A)))
sumlogdiag{T}(J::UniformScaling{T},d) = log(J.λ)*d
function logpdfnormal(x, A) 
    S = chol(A, Val{:L})
    d = length(x)
     -((norm(S\x))^2 + 2sumlogdiag(S,d) + d*log(2pi))/2
end
function logpdfnormal(x::Float64, A) 
    S = sqrt(A)
     -((norm(S\x))^2 + 2log(S) + log(2pi))/2
end

function trace{m,T}(A::Mat{m,m,T}) 
    t = zero(T)
    for i in 1:m
        t += A[i,i]
    end
    t
end    
