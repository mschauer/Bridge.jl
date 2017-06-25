# Gaussian
using Distributions
import Base: rand
import Distributions: pdf, logpdf

sumlogdiag(a::Float64, d=1) = log(a)
sumlogdiag(A,d) = sum(log.(diag(A)))
sumlogdiag{T}(J::UniformScaling{T},d) = log(J.λ)*d
 


import Distributions: logpdf, pdf
mutable struct Gaussian{T}
    mu::T
    a
    sigma
    Gaussian{T}(mu, a) where T = new(mu, a, chol(a)')
end
Gaussian{T}(mu::T, a) = Gaussian{T}(mu, a)

rand(P::Gaussian) = P.mu + P.sigma*randn(typeof(P.mu))
rand{T}(P::Gaussian{Vector{T}}) = P.mu + P.sigma*randn(T, length(P.mu))
function logpdf(P::Gaussian, x)
    S = P.sigma
    x = x - P.mu
    d = length(x)
     -((norm(S\x))^2 + 2sumlogdiag(S,d) + d*log(2pi))/2
end

pdf(P::Gaussian,x) = exp(logpdf(P::Gaussian, x))

function Base.LinAlg.chol(u::UniformScaling{T}) where T<:Real
    u.λ >= 0 || error(Base.LinAlg.PosDefException(1))
    return UniformScaling{T}(sqrt(u.λ))
end

function Base.LinAlg.chol(u::SDiagonal{N,T}) where T<:Real where N
    all(u.diag .>= zero(T)) || error(Base.LinAlg.PosDefException(1))
    return SDiagonal(sqrt.(u.diag))
end

"""
    logpdfnormal(x, A) 

logpdf of centered gaussian with covariance A
"""
function logpdfnormal(x, A) 

    S = chol((A+A')/2)'

    d = length(x)
     -((norm(S\x))^2 + 2sumlogdiag(S,d) + d*log(2pi))/2
end
function logpdfnormal(x::Float64, A) 
    S = sqrt(A)
     -((norm(S\x))^2 + 2log(S) + log(2pi))/2
end

