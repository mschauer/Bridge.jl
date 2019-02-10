
struct MarslandShardlow{T} <: ContinuousTimeProcess{State{Point}}  # used to be called Landmarks
    a::T # kernel std parameter
    γ::T # noise level
    λ::T # mean reversion
    n::Int
end

# specify auxiliary process
struct MarslandShardlowAux{T} <: ContinuousTimeProcess{State{Point}}
    a::T # kernel std parameter
    γ::T # noise level
    λ::T # mean reversion
    xT::State{Point}  # use x = State(P.v, zero(P.v)) where v is conditioning vector
    n::Int
end

MarslandShardlowAux(P::MarslandShardlow, v) = MarslandShardlowAux(P.a, P.γ, P.λ, v, P.n)

function Bridge.σ!(t, x, dm, out, P::Union{MarslandShardlow, MarslandShardlowAux})
    zero!(out.q)
    out.p .= dm*P.γ
    out
end

"""
Multiply a(t,x) times xin (which is of type state)
Returns variable of type State
"""
function amul(t, x, xin::State, P::Union{MarslandShardlow, MarslandShardlowAux})
    out = copy(xin)
    zero!(out.q)
    out.p .= P.γ^2 .* xin.p
    out
end

"""
Returns matrix a(t) for Marsland-Shardlow model
"""
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
#Bridge.a(t, x, P::Union{MarslandShardlow, MarslandShardlowAux}) = Bridge.a(t, P::Union{MarslandShardlow, MarslandShardlowAux})
Bridge.a(t, x, P::Union{MarslandShardlow, MarslandShardlowAux}) = Bridge.a(t, P)

Bridge.constdiff(::Union{MarslandShardlow, MarslandShardlowAux}) = true
