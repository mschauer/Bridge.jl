
struct MarslandShardlow{T} <: ContinuousTimeProcess{State{Point}}  # used to be called Landmarks
    a::T # kernel parameter
    γ::T # noise level
    λ::T # mean reversion
    #v::Vector{T}  # conditioning
    n::Int
end

# specify auxiliary process
struct MarslandShardlowAux{T} <: ContinuousTimeProcess{State{Point}}
    a::T # kernel parameter
    γ::T # noise level
    λ::T # mean reversion
    xT::State{Point}  # use x = State(P.v, zero(P.v)) where v is conditioning vector
    n::Int
end

MarslandShardlowAux(P::MarslandShardlow, v) = MarslandShardlowAux(P.a, P.γ, P.λ, v, P.n)

# Gaussian kernel
kernel(x, P::Union{MarslandShardlow, MarslandShardlowAux}) = 1/(2*π*P.a)^(length(x)/2)*exp(-norm(x)^2/(2*P.a))

zero!(v) = v[:] = fill!(v, zero(eltype(v)))

function hamiltonian((q, p), P)
    s = 0.0
    for i in eachindex(q), j in eachindex(q)
      #  i == j && continue
        s += 1/2*dot(p[i], p[j])*kernel(q[i] - q[j], P)
    end
    s
end

Bridge.b(t::Float64, x, P::Union{MarslandShardlow, MarslandShardlowAux}) = Bridge.b!(t, x, copy(x), P)
function Bridge.b!(t, x, out, P::MarslandShardlow)
    zero!(out)
    for i in 1:P.n
        for j in 1:P.n
            out.q[i] += 0.5*p(x,j)*kernel(q(x,i) - q(x,j), P)
            # heat bath
            out.p[i] += -P.λ*0.5*p(x,j)*kernel(q(x,i) - q(x,j), P) +
                1/(2*P.a) * dot(p(x,i), p(x,j)) * (q(x,i)-q(x,j))*kernel(q(x,i) - q(x,j), P)
        end
    end
    out
end

function Bridge.b!(t, x, out, P::MarslandShardlowAux)
    zero!(out)
    for i in 1:P.n
        for j in 1:P.n
            @inbounds out.q[i] += 0.5*p(x,j)*kernel(q(P.xT,i) - q(P.xT,j), P)
            # heat bath
            # out[posp(i)] += -P.λ*0.5*p(x,j)*kernel(q(x,i) - q(x,j), P) +
            #     1/(2*P.a) * dot(p(x,i), p(x,j)) * (q(x,i)-q(x,j))*kernel(q(x,i) - q(x,j), P)
        end
    end
    out
end

"""
Compute tildeB(t)
"""
function Bridge.B(t, P::MarslandShardlowAux)
    I = Int[]
    J = Int[]
    X = Unc[]
    for i in 1:P.n
        for j in 1:P.n
            push!(I, 2i - 1)
            push!(J, 2j)
            push!(X, 0.5*kernel(q(P.xT,i) - q(P.xT,j), P)*one(Unc))
        end
    end
    B = sparse(I, J, X, 2P.n, 2P.n)
end

# function Bridge.B!(t,x,out, P::MarslandShardlowAux)
#     I = Int[]
#     J = Int[]
#     X = Unc[]
#     for i in 1:P.n
#         for j in 1:P.n
#             push!(I, 2i - 1)
#             push!(J, 2j)
#             push!(X, 0.5*kernel(q(P.xT,i) - q(P.xT,j), P)*one(Unc))
#         end
#     end
#     out .= Matrix(sparse(I, J, X, 2P.n, 2P.n)) * x  # BAD: inefficient
# end

"""
Compute B̃(t) * X (B̃ from auxiliary process) and write to out
Both B̃(t) and X are of type UncMat
"""
function Bridge.B!(t,X,out, Paux::MarslandShardlowAux)
    out .= 0.0 * out
    for k in 1:2Paux.n # loop over out[p(i), p(k)]
        for i in 1:Paux.n
            for j in 1:Paux.n
                out[2i-1,k] += 0.5*X[p(j), k]*kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)
                #out[2i,k] += -Paux.λ*0.5*X[p(j), k]*kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)
             end
        end
    end
    out
end


function Bridge.σ!(t, x, dm, out, P::Union{MarslandShardlow, MarslandShardlowAux})
    zero!(out.q)
    out.p .= dm*P.γ
    out
end

"""
Multiply a(t,x) times xin (which is of type state)
Returns multiplication of type state
"""
function amul(t, x, xin::State, P::Union{MarslandShardlow, MarslandShardlowAux})
    out = copy(xin)
    zero!(out.q)
    out.p .= P.γ^2 .* xin.p
    out
end


function Bridge.a(t,  P::Union{MarslandShardlow, MarslandShardlowAux})
    I = Int[]
    X = Unc[]
    γ2 = P.γ^2
    for i in 1:P.n
            push!(I, 2i)
            push!(X, γ2*one(Unc))
    end
    sparse(I, I, X, 2P.n, 2P.n)
end


Bridge.a(t, x, P::Union{MarslandShardlow, MarslandShardlowAux}) = Bridge.a(t, P::Union{MarslandShardlow, MarslandShardlowAux})

# """
# Replacing out with dP, which is  B*arg + arg*B'- tilde_a
# """
# function Bridge.dP!(t, p, out, P::MarslandShardlowAux)
#     Bridge.B!(t, p, out, P)
#     out .= out .+ out'
#     γ2 = P.γ^2
#     for i in 1:P.n
#         out[2i, 2i] -= γ2*one(Unc)  # according to ordering position, momentum ,position, momentum
#     end
#     out
# end
#
# function dP!(t, p, out, P)
#     B!(t, p, out, P)
#     out .= out .+ out' - a(t, P)
#     out
# end

Bridge.constdiff(::Union{MarslandShardlow, MarslandShardlowAux}) = true
