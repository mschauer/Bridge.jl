module LowrankRiccati

#=
include("LowrankRiccati.jl")
using .LowrankRiccati
=#
export lowrankriccati, lowrankriccati!

using SparseArrays
using SuiteSparse
using LinearAlgebra
using Expokit  # for matrix exponentials
using BenchmarkTools
using Distributions


# we use Expokit, which gives the funciton expmv!
# w = expmv!{T}( w::Vector{T}, t::Number, A, v::Vector{T}; kwargs...)
# The function expmv! calculates w = exp(t*A)*v, where A is a matrix or any type that supports size, eltype and mul! and v is a dense vector by using Krylov subspace projections. The result is stored in w.

"""
Compute exp(A τ) * U
"""
function expmat(τ,A,U)
    M = []
    for j in 1:size(U,2)
       push!(M, expmv!(zeros(size(U,1)),τ,A,U[:,j]))
    end
    hcat(M...)
end

"""
 one step of low rank ricatti
 implement Mena et al. for solving
 (d P)/(d t) = A P(t) + P(t) A' + Q,   P(t_0) = P0
 with a low rank approximation for P
"""
function lowrankriccati!(s, t, A, Q, (S,U), (Sout, Uout))
    τ = t-s # time step
    Uᴬ, R = qr!(expmat(τ,A,U))
    Uᴬ = Matrix(Uᴬ) # convert from square to tall matrix
    # step 4
    Sᴬ = R * S * R'
    # step 5a (gives new U)
    U, Ŝ = qr!(Uᴬ * Sᴬ + (τ * Q) * Uᴬ)
    Uout .= Matrix(U)
    # step 5b
    Ŝ = Ŝ - Uout' * (τ * Q) * Uᴬ
    # step 5c (gives new S)
    L = Ŝ * Uᴬ' +  Uout' * (τ * Q)
    Sout .= L * Uout

    Sout, Uout
end
lowrankriccati(s, t, A, Q, (S, U)) = lowrankriccati!(s, t, A, Q, (S, U), (copy(S), copy(U)))

using Test
@testset "low rank riccati" begin

    # Test problem
    d = 30
    Q = sprand(d,d,.01) - I# Diagonal(randn(d))
    Q = Q * Q'
    A = sprand(d,d,0.1)-I #Diagonal(diagels)
    diagels = 0.4.^(1:d)
    P0 =  Diagonal(diagels)

    ld = 30 # low rank dimension

    s = 0.3
    t = 0.31
    τ = t - s

    # step 1
    M0 = eigen(Matrix(P0))
    S = Matrix(Diagonal(M0.values[1:ld]))
    U = M0.vectors[:,1:ld]


    # Low rank riccati solution to single time step
    S, U = lowrankriccati(s, t, A, Q, (S, U))
    P = U * S * U'

    # Compute exact solution
    λ = lyap(Matrix(A), -Matrix(Q)) + P0
    Φ = exp(Matrix(A)*τ)

    Pexact = Φ*λ*Φ' - λ + P0
    # assess difference
    @test norm(P - Pexact) < 0.001
end

end
