#### Landmarks specification

struct MarslandShardlow{T} <: ContinuousTimeProcess{State{Point}}
    a::T # kernel std parameter
    γ::T # noise level
    λ::T # mean reversion
    n::Int
end

struct MarslandShardlowAux{T} <: ContinuousTimeProcess{State{Point}}
    a::T # kernel std parameter
    γ::T # noise level
    λ::T # mean reversion
    xT::State{Point}  # use x = State(P.v, zero(P.v)) where v is conditioning vector
    n::Int
end

MarslandShardlowAux(P::MarslandShardlow, xT) = MarslandShardlowAux(P.a, P.γ, P.λ, xT, P.n)

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


"""
kernel in Hamiltonian
"""
function kernel(q, P::Union{Landmarks, LandmarksAux,
MarslandShardlow, MarslandShardlowAux})
 (2*π*P.a^2)^(-d/2)*exp(-norm(q)^2/(2*P.a^2))
end

function ∇kernel(q, P::Union{Landmarks, LandmarksAux, MarslandShardlow, MarslandShardlowAux})
    -P.a^(-2) * kernel(q,P) * q
end

"""
Needed for b! in case P is auxiliary process
"""
function ∇kernel(q,qT, P::Union{Landmarks, LandmarksAux, MarslandShardlow, MarslandShardlowAux})
     -P.a^(-2) * kernel(qT,P) * q
end


function hamiltonian((q, p), P)
    s = 0.0
    for i in eachindex(q), j in eachindex(q)
        s += 1/2*dot(p[i], p[j])*kernel(q[i] - q[j], P)
    end
    s
end



# kernel for noisefields
function K̄(q,τ)
     exp(-norm(q)^2/(2*τ^2))# (2*π*τ^2)^(-d/2)*exp(-norm(x)^2/(2*τ^2))
end
# gradient of kernel for noisefields
function ∇K̄(q,τ)
     -τ^(-2) * K̄(q,τ) * q
end
"""
Needed for b! in case P is auxiliary process
"""
function ∇K̄(q, qT, τ)
    -τ^(-2) * K̄(qT,τ) * q
end

# function for specification of diffusivity of landmarks
σq(q, nf::Noisefield) = diagm(0 =>nf.λ) * K̄(q - nf.δ,nf.τ)
σp(q, p, nf::Noisefield) = -diagm(0 => p.*nf.λ.* ∇K̄(q - nf.δ,nf.τ))


function σq(x::Point, nfs::Array{Noisefield,1})
    out = 0.0 * x
    for j in 1:length(nfs)
        out += σq(x, nfs[j])
    end
    out
end

Bridge.b(t::Float64, x, P::Union{Landmarks,MarslandShardlow})= Bridge.b!(t, x, copy(x), P)
Bridge.b(t::Float64, x, P::Union{LandmarksAux,MarslandShardlowAux})= Bridge.b!(t, x, copy(x), P)

"""
Evaluate drift of landmarks in (t,x) and save to out
x is a state and out as well
"""
function Bridge.b!(t, x, out, P::MarslandShardlow)
    zero!(out)
    for i in 1:P.n
        for j in 1:P.n
            out.q[i] += p(x,j)*kernel(q(x,i) - q(x,j), P)
            # heat bath
            out.p[i] += -P.λ*p(x,j)*kernel(q(x,i) - q(x,j), P) -
                 dot(p(x,i), p(x,j)) * ∇kernel(q(x,i) - q(x,j), P)
        end
    end
    out
end

"""
Evaluate drift of landmarks in (t,x) and save to out
x is a state and out as well
"""
function Bridge.b!(t, x, out, P::Landmarks)
    zero!(out)
    for i in 1:P.n
        for j in 1:P.n
            out.q[i] += p(x,j)*kernel(q(x,i) - q(x,j), P)
            out.p[i] +=  -dot(p(x,i), p(x,j)) * ∇kernel(q(x,i) - q(x,j), P)
        end
        if itostrat
            global ui = zero(Unc)
            for k in 1:length(P.nfs)
                nf = P.nfs[k]
                ui += 0.5 * nf.τ^(-2) * diagm(0 =>nf.λ.^2) * K̄(q(x,i)-nf.δ,nf.τ)^2
            end
            out.q[i] -=  ui * q(x,i)
            out.p[i] += ui * p(x,i)
        end
    end
    out
end


"""
Evaluate drift of landmarks auxiliary process in (t,x) and save to out
x is a state and out as well
"""
function Bridge.b!(t, x, out, Paux::MarslandShardlowAux)
    zero!(out)
    for i in 1:Paux.n
        for j in 1:Paux.n
            out.q[i] += p(x,j)*kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)
            out.p[i] +=   -Paux.λ*p(x,j)*kernel(q(Paux.xT,i) - q(Paux.xT,j), P)
        end
    end
    out
end

"""
Evaluate drift of landmarks auxiliary process in (t,x) and save to out
x is a state and out as well
"""
function Bridge.b!(t, x, out, Paux::LandmarksAux)
    zero!(out)
    for i in 1:Paux.n
        for j in 1:Paux.n
            out.q[i] += p(x,j)*kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)
        end
            # out.p[i] +=   -P.λ*0.5*p(x,j)*kernel(q(P.xT,i) - q(P.xT,j), P)
            #     -0.5* dot(p(P.xT,i), p(P.xT,j)) * ∇kernel(q(x,i) - q(x,j),q(P.xT,i) - q(P.xT,j), P)
        if itostrat
            global ui = zero(Unc)
            for k in 1:length(P.nfs)
                nf = P.nfs[k]
                ui = 0.5 * nf.τ^(-2) * diagm(0 =>nf.λ.^2) * K̄(q(Paux.xT,i)-nf.δ,nf.τ)^2
            end
            out.q[i] -=  ui * q(x,i)
            out.p[i] += ui * p(x,i)
        end
    end
    out
end


"""
Compute tildeB(t) for landmarks auxiliary process
"""
function Bridge.B(t, Paux::MarslandShardlowAux)
    Iind = Int[]
    Jind = Int[]
    X = Unc[]
    for i in 1:Paux.n
        for j in 1:Paux.n
            # terms for out.q[i]
            push!(Iind, 2i - 1)
            push!(Jind, 2j)
            push!(X, kernel(q(Paux.xT,i) - q(Paux.xT,j), P)*one(Unc))

            push!(Iind,2i)
            push!(Jind,2j)
            push!(X,  -Paux.λ*kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)*one(Unc))
        end
    end
    sparse(Iind, Jind, X, 2Paux.n, 2Paux.n)
end

"""
Compute tildeB(t) for landmarks auxiliary process
"""
function Bridge.B(t, Paux::LandmarksAux)
    Iind = Int[]
    Jind = Int[]
    X = Unc[]
    for i in 1:Paux.n
        for j in 1:Paux.n
            # terms for out.q[i]
            push!(Iind, 2i - 1)
            push!(Jind, 2j)
            push!(X, kernel(q(Paux.xT,i) - q(Paux.xT,j), P)*one(Unc))

            if itostrat
                global out1 = zero(Unc)
                for k in 1:length(Paux.nfs)
                    nf = P.nfs[k]
                    out1 -= 0.5 * nf.τ^(-2) * Unc(diagm(0 =>nf.λ.^2)) *  K̄(q(Paux.xT,i)-nf.δ,nf.τ)^2
                end
                push!(Iind, 2i - 1)
                push!(Jind, 2j - 1)
                push!(X, out1)
                push!(Iind, 2i)
                push!(Jind, 2j)
                push!(X, -out1)
            end

            # terms for out.p[i]
            # push!(I, 2i)
            # push!(J, 2j-1)
            # if j==i
            #     push!(X, sum([1/(2*Paux.a) * dot(p(Paux.xT,i), p(Paux.xT,j)) * kernel(q(Paux.xT,i) - q(Paux.xT,j), P)  for j in setdiff(1:Paux.n,i)]) * one(Unc))
            # else
            #     push!(X, -1/(2*Paux.a) * dot(p(Paux.xT,i), p(Paux.xT,j)) * kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)*one(Unc))
            # end
        end
    end
    sparse(Iind, Jind, X, 2Paux.n, 2Paux.n)
end


"""
Compute B̃(t) * X (B̃ from auxiliary process) and write to out
Both B̃(t) and X are of type UncMat
"""
function Bridge.B!(t,X,out, Paux::MarslandShardlowAux)
    out .= 0.0 * out
    for i in 1:Paux.n  # separately loop over even and odd indices
        for k in 1:2Paux.n # loop over all columns
            for j in 1:Paux.n
                 out[2i-1,k] += kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)*X[p(j), k] #*one(Unc)
                 out[2i,k] += -Paux.λ*kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)*one(Unc)*X[p(j), k] #*one(Unc)
                # if j==i
                #    out[2i,k] +=  X[q(j),k] *sum([1/(2*Paux.a) * dot(p(Paux.xT,i), p(Paux.xT,j)) *
                #                         kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)  for j in setdiff(1:Paux.n,i)])*one(Unc)
                # else
                #    out[2i,k] +=  X[q(j),k] *(-1/(2*Paux.a)) *  dot(p(Paux.xT,i), p(Paux.xT,j)) * kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)*one(Unc)
                # end
            end
        end
    end

    out
end

"""
Compute B̃(t) * X (B̃ from auxiliary process) and write to out
Both B̃(t) and X are of type UncMat
"""
function Bridge.B!(t,X,out, Paux::LandmarksAux)
#### HAS NOT BEEN IMPLEMENTED FOR ITO-STRAT CORRECTION (probably doesn't pay off the effort)
    out .= 0.0 * out
    quickway = true
    if quickway==true
        out .= Matrix(Bridge.B(t,Paux)) * X
    else # still need to implement an efficient way here
        for i in 1:Paux.n  # separately loop over even and odd indices
            for k in 1:2Paux.n # loop over all columns
                for j in 1:Paux.n
                     out[2i-1,k] += kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)*X[p(j), k] #*one(Unc)
                    # if j==i
                    #    out[2i,k] +=  X[q(j),k] *sum([1/(2*Paux.a) * dot(p(Paux.xT,i), p(Paux.xT,j)) *
                    #                         kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)  for j in setdiff(1:Paux.n,i)])*one(Unc)
                    # else
                    #    out[2i,k] +=  X[q(j),k] *(-1/(2*Paux.a)) *  dot(p(Paux.xT,i), p(Paux.xT,j)) * kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)*one(Unc)
                    # end
                end
            end
        end
    end
    out
end

function Bridge.β(t, Paux::Union{MarslandShardlowAux,LandmarksAux})
    State(zeros(Point,Paux.n), zeros(Point,Paux.n))
end


if TEST
    Hend⁺ = [rand(Unc) for i in 1:2Paux.n, j in 1:2Paux.n]
    t0 = 2.0
    BB = Matrix(Bridge.B(t0,Paux)) * Hend⁺
    out = deepcopy(Hend⁺)
    Bridge.B!(t0,Hend⁺,out,Paux)
    @test out==BB
end



function Bridge.σ!(t, x, dm, out, P::Union{MarslandShardlow, MarslandShardlowAux})
    zero!(out.q)
    out.p .= dm*P.γ
    out
end


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


"""
Compute sigma(t,x)' * y where y is a state and sigma is the diffusion coefficient of landmarks
returns a vector of points of length P.nfs
"""
function σtmul(t, x_, y::State, P::Union{Landmarks,LandmarksAux})
    if P isa Landmarks
        x = x_
    else
        x = P.xT
    end
    out = zeros(Point, length(P.nfs))
    for j in 1:length(P.nfs)
        for i in 1:P.n
            out[j] += σq(q(x, i), P.nfs[j])' * y.q[i] +
                        σp(q(x, i), p(x, i), P.nfs[j])' * y.p[i]
        end
    end
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


# function Bridge.a(t, x_, P::Union{Landmarks,LandmarksAux})
#     if P isa Landmarks
#         x = x_
#     else
#         x = P.xT
#     end
#     out = zeros(Unc,2P.n,2P.n)
#     for i in 1:P.n
#         for k in 1:P.n
#             for j in 1:length(P.nfs)
#                 out[2i-1,2k-1] += σq(q(x,i),P.nfs[j]) * σq(q(x, k),P.nfs[j])'
#                 out[2i-1,2k] += σq(q(x,i),P.nfs[j]) * σp(q(x,k),p(x,k),P.nfs[j])'
#                 out[2i,2k-1] += σp(q(x,i),p(x,i),P.nfs[j]) * σq(q(x,k),P.nfs[j])'
#                 out[2i,2k] += σp(q(x,i),p(x,i),P.nfs[j]) * σp(q(x,k),p(x,k),P.nfs[j])'
#             end
#         end
#     end
#     out
# end

function Bridge.a(t, x_, P::Union{Landmarks,LandmarksAux})
    if P isa Landmarks
        x = x_
    else
        x = P.xT
    end
    out = zeros(Unc,2P.n,2P.n)
    for i in 1:P.n
        for k in i:P.n
            for j in 1:length(P.nfs)
                out[2i-1,2k-1] += σq(q(x,i),P.nfs[j]) * σq(q(x, k),P.nfs[j])'
                out[2i-1,2k] += σq(q(x,i),P.nfs[j]) * σp(q(x,k),p(x,k),P.nfs[j])'
                out[2i,2k-1] += σp(q(x,i),p(x,i),P.nfs[j]) * σq(q(x,k),P.nfs[j])'
                out[2i,2k] += σp(q(x,i),p(x,i),P.nfs[j]) * σp(q(x,k),p(x,k),P.nfs[j])'
            end
      end
    end
    for i in 2:2P.n
        for k in 1:i-1
            out[i,k] = out[k,i]
        end
    end
    out
end


Bridge.a(t, P::LandmarksAux) =  Bridge.a(t, 0, P)
#Bridge.a(t, P::Union{Landmarks,LandmarksAux}) =  Bridge.a(t, P.xT, P::Union{Landmarks,LandmarksAux})

"""
Multiply a(t,x) times xin (which is of type state)
Returns variable of type State
"""
function amul(t, x::State, xin::State, P::Union{MarslandShardlow, MarslandShardlowAux})
    out = copy(xin)
    zero!(out.q)
    out.p .= P.γ^2 .* xin.p
    out
end
function amul(t, x::State, xin::Vector{Point}, P::Union{MarslandShardlow, MarslandShardlowAux})
    out = copy(x)
    zero!(out.q)
    out.p .= P.γ^2 .* vecofpoints2state(xin).p
    out
end


"""
Multiply a(t,x) times a vector of points
Returns a State
(first multiply with sigma', via function σtmul, next left-multiply this vector with σ)
"""
function amul(t, x::State, xin::Vector{Point}, P::Union{Landmarks,LandmarksAux})
    #vecofpoints2state(Bridge.a(t, x, P)*xin)
    out = copy(x)
    zero!(out)
    Bridge.σ!(t, x, σtmul(t, x, vecofpoints2state(xin), P),out,P)
end

"""
Multiply a(t,x) times a state
Returns a state
(first multiply with sigma', via function σtmul, next left-multiply this vector with σ)
"""
function amul(t, x::State, xin::State, P::Union{Landmarks,LandmarksAux})
    #vecofpoints2state(Bridge.a(t, x, P)*vec(xin))
    out = copy(x)
    zero!(out)
    Bridge.σ!(t, x, σtmul(t, x, xin, P),out,P)
end



Bridge.constdiff(::Union{MarslandShardlow, MarslandShardlowAux,LandmarksAux}) = true
Bridge.constdiff(::Landmarks) = false
