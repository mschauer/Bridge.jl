
"""
Specify parameters for noisefield
λ * exp(-|δ - q|^2/τ^2)
"""
struct Noisefield
    δ::Point   # locations of noise field
    λ::Point   # scaling at noise field
    τ::Float64 # std of Gaussian kernel noise field
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
LandmarksAux(P::Landmarks, xT) = LandmarksAux(P.a, P.λ, xT, P.n, P.nfs)

# Gaussian kernel
kernel(x, P::Union{Landmarks, LandmarksAux}) = 1/(2*π*P.a)^(length(x)/2)*exp(-norm(x)^2/(2*P.a))

# Kernel of the noise fields (need these functions to evaluate diffusion coefficient in ahs model)
noiseσ(q, nf::Noisefield) = nf.λ * exp(-norm(nf.δ - q)^2/nf.τ^2)
eta(q, p, nf::Noisefield) = dot(p, nf.λ)/nf.τ^2 * (q - nf.δ) * exp(-norm(nf.δ - q)^2/nf.τ^2)

function hamiltonian((q, p), P)
    s = 0.0
    for i in eachindex(q), j in eachindex(q)
      #  i == j && continue
        s += 1/2*dot(p[i], p[j])*kernel(q[i] - q[j], P)
    end
    s
end

Bridge.b(t::Float64, x, P::Union{Landmarks, LandmarksAux}) = Bridge.b!(t, x, copy(x), P)

"""
Evaluate drift of landmarks in (t,x) and save to out
x is a state and out as well
"""
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

"""
Evaluate drift of landmarks auxiliary process in (t,x) and save to out
x is a state and out as well
"""
function Bridge.b!(t, x, out, P::LandmarksAux)
    zero!(out)
    for i in 1:P.n
        for j in 1:P.n
            out.q[i] += 0.5*p(x,j)*kernel(q(P.xT,i) - q(P.xT,j), P)
            # heat bath
            out.p[i] +=# -P.λ*0.5*p(x,j)*kernel(q(x,i) - q(x,j), P) +
                 1/(2*P.a) * dot(p(P.xT,i), p(P.xT,j)) * (q(x,i)-q(x,j))*kernel(q(P.xT,i) - q(P.xT,j), P)
        end
    end
    out
end


"""
Compute tildeB(t) for landmarks auxiliary process
"""
function Bridge.B(t, Paux::LandmarksAux)
    I = Int[]
    J = Int[]
    X = Unc[]
    for i in 1:Paux.n
        for j in 1:Paux.n
            # terms for out.q[i]
            push!(I, 2i - 1)
            push!(J, 2j)
            push!(X, 0.5*kernel(q(Paux.xT,i) - q(Paux.xT,j), P)*one(Unc))

            # terms for out.p[i]
            push!(I, 2i)
            push!(J, 2j-1)
            if j==i
                push!(X, sum([1/(2*Paux.a) * dot(p(Paux.xT,i), p(Paux.xT,j)) * kernel(q(Paux.xT,i) - q(Paux.xT,j), P)  for j in setdiff(1:Paux.n,i)]) * one(Unc))
            else
                push!(X, -1/(2*Paux.a) * dot(p(Paux.xT,i), p(Paux.xT,j)) * kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)*one(Unc))
            end
        end
    end
    B = sparse(I, J, X, 2Paux.n, 2Paux.n)
end

# function Bridge.B!(t,X,out, P::LandmarksAux)
#     I = Int[]
#     J = Int[]
#     B = Unc[]
#     for i in 1:P.n
#         for j in 1:P.n
#             push!(I, 2i - 1)
#             push!(J, 2j)
#             push!(B, 0.5*kernel(q(P.xT,i) - q(P.xT,j), P)*one(Unc))
#         end
#     end
#     out .= Matrix(sparse(I, J, B, 2P.n, 2P.n)) * X  # BAD: inefficient
# end
"""
Compute B̃(t) * X (B̃ from auxiliary process) and write to out
Both B̃(t) and X are of type UncMat
"""
function Bridge.B!(t,X,out, Paux::LandmarksAux)
    out .= 0.0 * out
    for i in 1:Paux.n  # separately loop over even and odd indices
        for k in 1:2Paux.n # loop over all columns
            for j in 1:Paux.n
                out[2i-1,k] += X[p(j), k] *0.5*kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux) *one(Unc)
                if j==i
                   out[2i,k] +=  X[q(j),k] *sum([1/(2*Paux.a) * dot(p(Paux.xT,i), p(Paux.xT,j)) *
                                        kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)  for j in setdiff(1:Paux.n,i)])*one(Unc)
                else
                   out[2i,k] +=  X[q(j),k] *(-1/(2*Paux.a)) *  dot(p(Paux.xT,i), p(Paux.xT,j)) * kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)*one(Unc)
                end
            end
        end
    end
    out
end

 #  A = reshape([Unc(rand(4)) for i in 1:6, j in 1:6],6,6))
 # out = copy(A)
 # Bridge.B!(1.0,A,out,Paux)

"""
Compute sigma(t,x) * dm where dm is a vector and sigma is the diffusion coefficient of landmarks
write to out which is of type State
"""
function Bridge.σ!(t, x, dm, out, P::Landmarks)
    zero!(out)
    nfs = P.nfs
    for i in 1:P.n
        #for (nf, dm) in zip(P.nfs, dm)
        for j in 1:length(nfs)
            #out.p[i] += noiseσ(q(x, i), nf) * dm
            #out.q[i] += eta(q(x, i), p(x, i), nf) * dm
            out.q[i] += noiseσ(q(x, i), nfs[j]) * dm[j]
            out.p[i] += eta(q(x, i), p(x, i), nfs[j]) * dm[j]
        end
    end
    out
end

"""
Compute tildesigma(t,x) * dm where dm is a vector and sigma is the diffusion coefficient of landmarksaux
write to out which is of type State
"""
function Bridge.σ!(t, x, dm, out, P::LandmarksAux)
    Bridge.σ!(t, P.xT, dm, out, P::Landmarks)
end




# function Bridge.a(t,  P::Union{MarslandShardlow, MarslandShardlowAux})
#     I = Int[]
#     X = Unc[]
#     γ2 = P.γ^2
#     for i in 1:P.n
#             push!(I, 2i)
#             push!(X, γ2*one(Unc))
#     end
#     a = sparse(I, I, X, 2P.n, 2P.n)
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
                r1 =  noiseσ(q(x,i),P.nfs[j]) * noiseσ(q(x, k),P.nfs[j])'
                r2 =  noiseσ(q(x,i),P.nfs[j]) * eta(q(x,k),p(x,k),P.nfs[j])'
                r3 =  eta(q(x,i),p(x,i),P.nfs[j]) * eta(q(x,k),p(x,k),P.nfs[j])'
                out[2i-1,2k-1] += r1
                out[2i,2k-1] += r2
                out[2i-1,2k] += r2
                out[2i,2k] += r3

            end
        end
    end
    out
end

Bridge.a(t, P::LandmarksAux) =  Bridge.a(t, P.xT, P)

"""
Multiply a(t,x) times in (which is of type state)
Returns multiplication of type state
"""
function amul(t, x, in::State, P)
    vecofpoints2state(Bridge.a(t, x, P)*vec(in))
end

Bridge.constdiff(::Landmarks) = false
Bridge.constdiff(::LandmarksAux) = true
