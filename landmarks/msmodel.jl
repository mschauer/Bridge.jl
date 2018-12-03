include("state.jl")

const d = 2
const Point = SArray{Tuple{d},Float64,1,d}       # point in R2
const Unc = SArray{Tuple{d,d},Float64,d,d*d}     # Matrix presenting uncertainty



#########
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
kernel(x, P) = 1/(2*π*P.a)^(length(x)/2)*exp(-norm(x)^2/(2*P.a))



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


function Bridge.σ!(t, x, dm, out, P::Union{MarslandShardlow, MarslandShardlowAux})
    zero!(out.q)
    out.p .= dm*P.γ
    out
end

Bridge.constdiff(::Union{MarslandShardlow, MarslandShardlowAux}) = true


if false

#=
function Bridge.a(t, P::Union{MarslandShardlow, MarslandShardlowAux})
    out = zeros(4*P.n,4*P.n)
    for i in (2*P.n+1): 4*P.n
        out[i,i] = 1.0
    end
    out * P.γ^2
end
Bridge.a(t, x, P::Union{MarslandShardlow, MarslandShardlowAux}) = Bridge.a(t, P::Union{MarslandShardlow, MarslandShardlowAux})
=#



function BBt!(t, arg, out, P::MarslandShardlowAux)
    B = zeros(4 * P.n,4 * P.n)
    for i in 1:P.n,  j in 1:P.n
            B[posq(i),posp(j)] = 0.5 * kernel(q(P.v,i)-q(P.v,j),P) * Matrix(1.0I,2,2)
    end
    out .= (B*arg + arg*B')
    out
end

# nog aan te passen
function Bridge.dP!(t, p, out, P)
    BBt!(t, p, out, P)
    for i in (2*P.n+1): 4*P.n
        out[i,i] -= P.γ^2
    end
    out
end


end
