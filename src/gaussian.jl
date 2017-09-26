# Gaussian
using Distributions
import Base: rand
import Distributions: pdf, logpdf

sumlogdiag(a::Float64, d=1) = log(a)
sumlogdiag(A,d) = sum(log.(diag(A)))
sumlogdiag(J::UniformScaling, d)= log(J.λ)*d
 
_logdet(A, d) = logdet(A)
_logdet(J::UniformScaling, d) = log(J.λ) * d

_symmetric(A) = Symmetric(A)
_symmetric(J::UniformScaling) = J

import Distributions: logpdf, pdf
mutable struct Gaussian{T}
    mu::T
    a
    sigma
    Gaussian{T}(mu, a) where T = new(mu, a, chol(a)')
end
Gaussian(mu::T, a) where {T} = Gaussian{T}(mu, a)

rand(P::Gaussian) = P.mu + P.sigma*randn(typeof(P.mu))
rand(P::Gaussian{Vector{T}}) where {T} = P.mu + P.sigma*randn(T, length(P.mu))
function logpdf(P::Gaussian, x)
    S = P.sigma
    x = x - P.mu
    d = length(x)
     -((norm(S\x))^2 + 2sumlogdiag(S,d) + d*log(2pi))/2
end

pdf(P::Gaussian,x) = exp(logpdf(P::Gaussian, x))

function Base.LinAlg.chol(u::SDiagonal{N,T}) where T<:Real where N
    all(u.diag .>= zero(T)) || error(Base.LinAlg.PosDefException(1))
    return SDiagonal(sqrt.(u.diag))
end

"""
    logpdfnormal(x, A) 

logpdf of centered Gaussian with covariance A
"""
function logpdfnormal(x, A) 

    S = chol(_symmetric(A))'

    d = length(x)
     -((norm(S\x))^2 + 2sumlogdiag(S,d) + d*log(2pi))/2
end
function logpdfnormal(x::Float64, a) 
     -(x^2/a + log(a) + log(2pi))/2
end

"""
logpdfnormalprec(x, A) 

logpdf of centered gaussian with precision A
"""
function logpdfnormalprec(x, A) 
    d = length(x)
    -(dot(x, S*x) - _logdet(A, d) + d*log(2pi))/2
end
logpdfnormalprec(x::Float64, a) =  -(a*x^2 - log(a) + log(2pi))/2
