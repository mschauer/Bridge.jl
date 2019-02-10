"""
Specify parameters for noisefield
"""
struct Noisefield
    δ::Point   # locations of noise field
    λ::Point   # scaling at noise field
    τ::Float64 # std of Gaussian kernel noise field
end

struct  Landmarks{T} <: ContinuousTimeProcess{State{Point}}
    a::T # kernel std
    λ::T # mean reversion
    n::Int64   # numer of landmarks
    nfs::Vector{Noisefield}  # vector containing pars of noisefields
end

struct LandmarksAux{T} <: ContinuousTimeProcess{State{Point}}
    a::T # kernel std
    λ::T # mean reversion
    xT::State{Point}  # use x = State(P.v, zero(P.v)) where v is conditioning vector
    n::Int64   # numer of landmarks
    nfs::Vector{Noisefield}  # vector containing pars of noisefields
end

LandmarksAux(P::Landmarks, xT) = LandmarksAux(P.a, P.λ, xT, P.n, P.nfs)

# kernel for noisefields
K̄(x,τ) = (2*π*τ^2)^(-d/2)*exp(-norm(x)^2/(2*τ^2))
# gradient of kernel for noisefields
∇K̄(x,τ) = -τ^(-2) * K̄(x,τ) * x
# function for specification of diffusivity of landmarks
σq(q, nf::Noisefield) = nf.λ * K̄(q - nf.δ,nf.τ)
σp(q, p, nf::Noisefield) = -dot(p, nf.λ) * ∇K̄(q - nf.δ,nf.τ)
# ∇q_σq(q, nf::Noisefield) = nf.λ * ∇K̄(q - nf.δ,nf.τ)
#
# function Δσq(q,nf::Noisefield,α::Int64)
#     out = Matrix{Float64}(undef,d,d) # would like to have Unc here
#     for β in 1:d
#         for γ in 1:d
#             out[β, γ] =  nf.τ^(-2) * nf.λ[β] * K̄(q,nf.τ) *
#                (nf.τ^(-2)*q[α]*q[γ] - 2* q[α]*(α==γ) )
#         end
#     end
#     out
# end

"""
Compute sigma(t,x) * dm where dm is a vector and sigma is the diffusion coefficient of landmarks
write to out which is of type State
"""
function Bridge.σ!(t, x_, dm, out, P::Union{Landmarks,LandmarksAux})
    if P isa Landmarks
        x = x_
    else
        x = P.xT
    end
    zero!(out)
    for i in 1:P.n
        for j in 1:length(P.nfs)
            out.q[i] += σq(q(x, i), P.nfs[j]) * dm[j]
            out.p[i] += σp(q(x, i), p(x, i), P.nfs[j]) * dm[j]
        end
    end
    out
end


# function Bridge.σ(t, x_, P::Union{Landmarks,LandmarksAux})
#     if P isa Landmarks
#         x = x_
#     else
#         x = P.xT
#     end
#     #zero!(out)
#     out = 0.0*Matrix{Unc}(undef,2P.n,length(P.nfs))
#     for i in 1:P.n
#         for j in 1:length(P.nfs)
#             out[2i-1,j] = σq(q(x, i), P.nfs[j])
#             out[2i,j] = σp(q(x, i), p(x, i), P.nfs[j])
#         end
#     end
#     out
# end


# """
# Compute tildesigma(t,x) * dm where dm is a vector and sigma is the diffusion coefficient of landmarksaux
# write to out which is of type State
# """
# function Bridge.σ!(t, x, dm, out, P::LandmarksAux)
#     Bridge.σ!(t, P.xT, dm, out, P::Landmarks)
# end

function Bridge.a(t, x_, P::Union{Landmarks,LandmarksAux})
    if P isa Landmarks
        x = x_
    else
        x = P.xT
    end
    out = zeros(Unc,2P.n,2P.n)
    for i in 1:P.n
        for k in 1:P.n
            for j in 1:length(P.nfs)
                # r1 =  σq(q(x,i),P.nfs[j]) * σq(q(x, k),P.nfs[j])'
                # r2 =  σq(q(x,i),P.nfs[j]) * σp(q(x,k),p(x,k),P.nfs[j])'
                # r3 =  σp(q(x,i),p(x,i),P.nfs[j]) * σp(q(x,k),p(x,k),P.nfs[j])'
                # out[2i-1,2k-1] += r1
                # out[2i-1,2k] += r2
                # out[2i,2k-1] += r2'
                # out[2i,2k] += r3
                r1 =  σq(q(x,i),P.nfs[j]) * σq(q(x, k),P.nfs[j])'
                r2 =  σq(q(x,i),P.nfs[j]) * σp(q(x,k),p(x,k),P.nfs[j])'
                r4 =  σp(q(x,i),p(x,i),P.nfs[j]) * σq(q(x,k),P.nfs[j])'
                r3 =  σp(q(x,i),p(x,i),P.nfs[j]) * σp(q(x,k),p(x,k),P.nfs[j])'
                out[2i-1,2k-1] += r1
                out[2i-1,2k] += r2
                out[2i,2k-1] += r4
                out[2i,2k] += r3
            end
        end
    end
    out
end

Bridge.a(t, P::LandmarksAux) =  Bridge.a(t, 0, P)
#Bridge.a(t, P::Union{Landmarks,LandmarksAux}) =  Bridge.a(t, P.xT, P::Union{Landmarks,LandmarksAux})

"""
Multiply a(t,x) times in (which is of type state)
Returns multiplication of type state
"""
function amul(t, x, in::State, P::Landmarks)
    vecofpoints2state(Bridge.a(t, x, P)*vec(in))
    #Bridge.a(t, x, P)*in
end


Bridge.constdiff(::Landmarks) = false
Bridge.constdiff(::LandmarksAux) = true
