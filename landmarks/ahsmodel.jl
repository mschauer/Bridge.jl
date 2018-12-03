include("state.jl")

const d = 2
const Point = SArray{Tuple{d},Float64,1,d}       # point in R2
const Unc = SArray{Tuple{d,d},Float64,d,d*d}     # Matrix presenting uncertainty

#########
struct Noisefield
    δ::Point   # locations of noise field
    λ::Point   # scaling at noise field
    τ::Float64 # variance of noise field
end

#########
struct  Landmarks{T} <: ContinuousTimeProcess{State{Point}}
    a::T # kernel parameter
    λ::T # mean reversion
    n::Int64   # numer of landmarks
    nfs::Vector{Noisefield}  # vector containing pars of noisefields
end

# specify auxiliary process
struct LandmarksAux{T} <: ContinuousTimeProcess{State{Point}}
    a::T # kernel parameter
    λ::T # mean reversion
    xT::State{Point}  # use x = State(P.v, zero(P.v)) where v is conditioning vector
    n::Int64   # numer of landmarks
    nfs::Vector{Noisefield}  # vector containing pars of noisefields
end
LandmarksAux(P::Landmarks, xT) = LandMarksAux(P.a, P.λ, xT, P.n, P.nfs)

# Gaussian kernel
kernel(x, P) = 1/(2*π*P.a)^(length(x)/2)*exp(-norm(x)^2/(2*P.a))

# Kernel of the noise fields (need these functions to evaluate diffusion coefficient in ahs model)
noiseσ(q, nf::Noisefield) = nf.λ * exp(-norm(nf.δ - q)^2/nf.τ)
eta(q, p, nf::Noisefield) = dot(p, nf.λ)/nf.τ * (q - nf.δ) * exp(-norm(nf.δ - q)^2/nf.τ)


zero!(v) = v[:] = fill!(v, zero(eltype(v)))
function hamiltonian((q, p), P)
    s = 0.0
    for i in eachindex(q), j in eachindex(q)
      #  i == j && continue
        s += 1/2*dot(p[i], p[j])*kernel(q[i] - q[j], P)
    end
    s
end

Bridge.b(t::Float64, x, P::Union{Landmarks, LandmarksAux}) = Bridge.b!(t, x, copy(x), P)
function Bridge.b!(t, x, out, P::Landmarks)
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

function Bridge.b!(t, x, out, P::LandmarksAux)
    zero!(out)
    for i in 1:P.n
        for j in 1:P.n
            out.q[i] += 0.5*p(x,j)*kernel(q(P.v,i) - q(P.v,j), P)
            # heat bath
            # out[posp(i)] += -P.λ*0.5*p(x,j)*kernel(q(x,i) - q(x,j), P) +
            #     1/(2*P.a) * dot(p(x,i), p(x,j)) * (q(x,i)-q(x,j))*kernel(q(x,i) - q(x,j), P)
        end
    end
    out
end

function Bridge.σ!(t, x, dm, out, P::Landmarks)
    zero!(out)
    nfs = P.nfs
    for i in 1:P.n
        #for (nf, dm) in zip(P.nfs, dm)
        for j in 1:length(nfs)
            #out.p[i] += noiseσ(q(x, i), nf) * dm
            #out.q[i] += eta(q(x, i), p(x, i), nf) * dm
            out.p[i] += noiseσ(q(x, i), nfs[j]) * dm[j]
            out.q[i] += eta(q(x, i), p(x, i), nfs[j]) * dm[j]
        end
    end
    out
end

function Bridge.σ!(t, x, dm, out, P::LandmarksAux)
    Bridge.σ!(t, P.xT, dm, out, P::Landmarks)
end

# function Bridge.a(t,x, P::Landmarks)
#     out = zeros(4*P.n,P.J)
#     for i in 1:P.n
#         for j in 1:P.J
#             out[posq(i),j] = noiseσ(q(x,i),P.nfs[j])
#             out[posp(i),j] = eta(q(x,i),p(x,i),P.nfs[j])
#         end
#     end
#     Bridge.outer(out)
# end
#
# Bridge.a(t,P::LandmarksAux) = Bridge.a(t, cat(P.v, zeros(2*P.n)),P::Landmarks )
# Bridge.a(t, x, P::LandmarksAux) = Bridge.a(t, P::LandmarksAux)
Bridge.constdiff(Landmarks) = false
Bridge.constdiff(LandmarksAux) = true
