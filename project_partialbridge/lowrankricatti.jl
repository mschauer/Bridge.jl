# implement Mena et al. for solving
# (d P)/(d t) = A P(t) + P(t) A' + Q,   P(t_0) = P0
# with a low rank approximation for Q

using SparseArrays
using SuiteSparse
using LinearAlgebra
using Expokit  # for matrix exponentials
using BenchmarkTools

# construct an example
d = 800
Q = sprand(d,d,.01) - I# Diagonal(randn(d))
diagels = 0.4.^(1:d)
A = sprand(d,d,0.1)#Diagonal(diagels)
P0 = Matrix(sprand(d,d,0.1))# Diagonal(diagels)

# we use Expokit, which gives the funciton expmv!
# w = expmv!{T}( w::Vector{T}, t::Number, A, v::Vector{T}; kwargs...)
# The function expmv! calculates w = exp(t*A)*v, where A is a matrix or any type that supports size, eltype and mul! and v is a dense vector by using Krylov subspace projections. The result is stored in w.

# It is much faster for large systems
v = rand(d)
@time a1=expmv!(zeros(d),1,Q,v)
@time a2=exp(Matrix(Q))*v
a1-a2

# Let's consider a few implementations
matexpvec1 = function(A,U,τ)
    exp(Matrix(τ * A)) * U
end

matexpvec2 = function(A,U,τ)
    M = []
    for j in 1:size(U,2)
       push!(M, expmv!(zeros(size(U,1)),τ,A,U[:,j]))
    end
    M = hcat(M...)
end

# matexpvec3 = function(A,U)  # is wrong for some reason
#     τ = 1.0
#     for j in 1:size(U,2)
#       expmv!(U[:,j],τ,A,U[:,j])
#     end
#     U
# end

matexpvec4 = function(A,U,τ)
    hcat([  expmv!(U[:,j],τ,A,U[:,j]) for  j in 1:size(U,2)]...)
end

τ = 1.0
R = A # or Q
M0 = svd(P0)
U = Matrix(M0.U[:,1:ld])

@benchmark  matexpvec1(R,U,τ)
@benchmark  matexpvec2(R,U,τ)
@benchmark  matexpvec4(R,U,τ)

maximum(matexpvec4(R,U,τ)-matexpvec1(R,U,τ))


function lowrankricatti(P0, A, Q, t, ld)
# P0 is the value of P at time t[1]; t is an increasing sequence of times,
# ld is the dimension of the low rank approximation
# output: at each time t a low rank approximation of P(t), where the couple (U,S) is such that P = U*S*U'

    # step 1
    M0 = svd(P0)
    global U = Matrix(M0.U[:,1:ld])
    global S = Diagonal(M0.S[1:ld])

    Ulowrank = [U]
    Slowrank = [S]
    for i in 1:length(t)-1
        τ = t[i+1] - t[i]
        if false
            matexp = exp(Matrix(τ * A))
            # step 3
            M = matexp * U
        else
            M = matexpvec4(A,U,τ)
        end
        Uᴬcompact, R = qr(M)
        Uᴬ = Matrix(Uᴬcompact)
        # step 4
        Sᴬ = R * S * R'
        # step 5
        K =  Uᴬ * Sᴬ + (τ * Q) * Uᴬ
        Ucompact, Ŝ = qr(K)
        U = Matrix(Ucompact)
        Ŝ = Ŝ - U' * (τ * Q) * Uᴬ
        L = Ŝ * Uᴬ' +  U' * (τ * Q)
        S = L * U

        push!(Ulowrank,U)
        push!(Slowrank,Diagonal(S))
        # ??? should S be diagonal or not: either convert it, or make Slowrank an array of full matrices
        #push!(Slowrank,S)
    end

    Ulowrank, Slowrank
end


# Example:
ld = 10  # low rank dimension
τ = 0.01; t0 =0.5; tend=0.8
t = t0:τ:tend

@time U,S = lowrankricatti(P0,A, Q, t,ld)
P = U[end]*S[end]*U[end]'

# compute exact solution
λ = lyap(Matrix(Q), -Matrix(A)) + P0
Φ = exp(Matrix(Q)*(tend- t0))
Pexact = Φ*λ*Φ' - λ + P0

# assess difference
dif = P - Pexact

print(maximum(abs.(dif)))
dif
