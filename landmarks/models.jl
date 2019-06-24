#### Landmarks specification
import Bridge: _b, _b!, B!, σ!, b!, σ, b

struct MarslandShardlow{T} <: ContinuousTimeProcess{State{PointF}}
    a::T # kernel std parameter
    γ::T # noise level
    λ::T # mean reversion
    n::Int
end

struct MarslandShardlowAux{S,T} <: ContinuousTimeProcess{State{PointF}}
    a::T # kernel std parameter
    γ::T # noise level
    λ::T # mean reversion
    xT::State{Point{S}}  # use x = State(P.v, zero(P.v)) where v is conditioning vector
    n::Int
end

MarslandShardlowAux(P::MarslandShardlow, xT) = MarslandShardlowAux(P.a, P.γ, P.λ, xT, P.n)

struct Noisefield{T}
    δ::Point{T}   # locations of noise field
    λ::Point{T}  # scaling at noise field
    τ::T # std of Gaussian kernel noise field
end

struct  Landmarks{S,T} <: ContinuousTimeProcess{State{PointF}}
    a::T # kernel std
    λ::T # mean reversion
    n::Int64   # numer of landmarks
    nfs::Vector{Noisefield{S}}  # vector containing pars of noisefields
end

struct LandmarksAux{S,T} <: ContinuousTimeProcess{State{PointF}}
    a::T # kernel std
    λ::T # mean reversion
    xT::State{Point{S}}  # use x = State(P.v, zero(P.v)) where v is conditioning vector
    n::Int64   # numer of landmarks
    nfs::Vector{Noisefield{S}}  # vector containing pars of noisefields
end

LandmarksAux(P::Landmarks, xT) = LandmarksAux(P.a, P.λ, xT, P.n, P.nfs)

const LandmarkModel = Union{Landmarks, LandmarksAux, MarslandShardlow, MarslandShardlowAux}

Bridge.constdiff(::Union{MarslandShardlow, MarslandShardlowAux,LandmarksAux}) = true
Bridge.constdiff(::Landmarks) = false


"""
kernel in Hamiltonian
"""
function kernel(q, P::LandmarkModel)
 (2*π*P.a^2)^(-d/2)*exp(-Bridge.inner(q)/(2*P.a^2))
end

"""
gradient of kernel in hamiltonian
"""
function ∇kernel(q, P::LandmarkModel)
    -P.a^(-2) * kernel(q, P) * q
end

"""
Needed for b! in case P is auxiliary process
"""
function ∇kernel(q, qT, P::LandmarkModel)
     -P.a^(-2) * kernel(qT, P) * q
end

"""
Hamiltonian for deterministic part of landmarks model
"""
function hamiltonian(x, P)
    s = 0.0
    for i in axes(x, 2), j in axes(x, 2)
#        s += 1/2*dot(p[i], p[j])*kernel(q[i] - q[j], P)
        s += 1/2*dot(x.p[i], x.p[j])*kernel(x.q[i] - x.q[j], P)
    end
    s
end

Bridge.b(t::Float64, x, P::Union{Landmarks,MarslandShardlow})= Bridge.b!(t, x, 0*x, P)

Bridge.σ(t, x, dm, P) =  Bridge.σ!(t, x, dm , 0*x, P)

########################################################################################################################################################################################
################ MS model #########################################################################################


"""
Evaluate drift of landmarks in (t,x) and save to out
x is a state and out as well
"""
function Bridge.b!(t, x, out, P::MarslandShardlow)
    zero!(out)
    for i in 1:P.n
        for j in 1:P.n
            out.q[i] += p(x,j)*kernel(q(x,i) - q(x,j), P)
            out.p[i] += -P.λ*p(x,j)*kernel(q(x,i) - q(x,j), P) -
                 dot(p(x,i), p(x,j)) * ∇kernel(q(x,i) - q(x,j), P)
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
            out.p[i] += -Paux.λ*p(x,j)*kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)
        end
    end
    out
end

"""
Compute tildeB(t) for landmarks auxiliary process
"""
function Bridge.B(t, Paux::MarslandShardlowAux)
    X = zeros(UncF, 2Paux.n, 2Paux.n)
    for i in 1:Paux.n
        for j in 1:Paux.n
            X[2i-1,2j] =  kernel(q(Paux.xT,i) - q(Paux.xT,j), P) * one(UncF)
            X[2i,2j] = -Paux.λ*kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)*one(UncF)
        end
    end
    X
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
                 out[2i-1,k] += kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux) * X[p(j), k]
                 out[2i,k] += -Paux.λ*kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)*one(UncF) * X[p(j), k]
            end
        end
    end
    out
end

function Bridge.β(t, Paux::MarslandShardlowAux)
    State(zeros(PointF,Paux.n), zeros(PointF,Paux.n))
end

function Bridge.σ!(t, x, dm, out, P::Union{MarslandShardlow, MarslandShardlowAux})
    zero!(out.q)
    out.p .= dm*P.γ
    out
end


"""
Returns matrix a(t) for Marsland-Shardlow model
"""
function Bridge.a(t,  P::Union{MarslandShardlow, MarslandShardlowAux})
    I = Int[]
    X = Unc{Float64}[]
    γ2 = P.γ^2
    for i in 1:P.n
            push!(I, 2i)
            push!(X, γ2*one(Unc{Float64}))
    end
    sparse(I, I, X, 2P.n, 2P.n)
end
Bridge.a(t, x, P::Union{MarslandShardlow, MarslandShardlowAux}) = Bridge.a(t, P)

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
function amul(t, x::State, xin::Vector{<:Point}, P::Union{MarslandShardlow, MarslandShardlowAux})
    out = copy(x)
    zero!(out.q)
    out.p .= P.γ^2 .* vecofpoints2state(xin).p
    out
end

########################################################################################################################################################################################
################ AHS model #########################################################################################
# kernel for noisefields
function K̄(q,τ)
     exp(-Bridge.inner(q)/(2*τ^2))# (2*π*τ^2)^(-d/2)*exp(-norm(x)^2/(2*τ^2))
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

"""
    Define z(q) = < ∇K̄(q - δ,τ), λ >
    Required for Stratonovich -> Ito correction in AHS-model
"""
z(q,τ,δ,λ) =  Bridge.inner(∇K̄(q - δ,τ),λ)

"""
    Define ∇z(q) = ∇ < ∇K̄(q - δ,τ), λ >
    Required for Stratonovich -> Ito correction in AHS-model
"""
∇z(q,τ,δ,λ) =  gradient(x -> z(x,τ,δ,λ),q)

# function for specification of diffusivity of landmarks
"""
    Suppose one noise field nf
    Returns diagonal matrix with noisefield for position at point location q (can be vector or Point)
"""
σq(q, nf::Noisefield) = Diagonal(nf.λ * K̄(q - nf.δ,nf.τ))

"""
    Suppose one noise field nf
    Returns diagonal matrix with noisefield for momentum at point location q (can be vector or Point)
"""
σp(q, p, nf::Noisefield) = -Diagonal(p .* nf.λ .* ∇K̄(q - nf.δ,nf.τ))


"""
    For AHS model compute total noise field on position experienced at a point x.
    Useful for plotting purposes.

    Example usage:
        σq(Point(0.0, 0.0), nfs)
        σq([0.0; 0.0], nfs)
"""
function σq(x, nfs::Array{<:Noisefield,1})
    out = σq(x, nfs[1])
        for j in 2:length(nfs)
            out += σq(x, nfs[j])
    end
    out
end

σq(nfs) = (x) -> σq(x,nfs)

"""
    Construct sequence of Noisefields for AHS model
    db: domainbound (sources are places on square grid specified by
        (-db:2nfstd:db) x -db:2nfstd:db
    nfstd: standard deviation of noise fields (the smaller: the more noise fields we use)
    γ: if set to one, then the value of the  noise field on the positions is approximately 1 at all locations in the domain
"""
function construct_nfs(db, nfstd, γ)
    r1 = -db:2nfstd:db
    r2 = -db:2nfstd:db
    nfloc = Point.(collect(product(r1, r2)))[:]
    nfscales = [2/pi*γ*Point(1.0, 1.0) for x in nfloc]  # intensity
    [Noisefield(δ, λ, nfstd) for (δ, λ) in zip(nfloc, nfscales)]
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
             for k in 1:length(P.nfs)
                 nf = P.nfs[k]
                 out.q[i] += 0.5 * z(q(x,i),nf.τ,nf.δ,nf.λ) * K̄(q(x,i)-nf.δ,nf.τ) * nf.λ
                 out.p[i] += 0.5 * dot(p(x,i),nf.λ) * ( z(q(x,i),nf.τ,nf.δ,nf.λ) * ∇K̄(q(x,i)-nf.δ,nf.τ) -K̄(q(x,i)-nf.δ,nf.τ) * ∇z(q(x,i),nf.τ,nf.δ,nf.λ) )
             end
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
        if itostrat
            for k in 1:length(P.nfs)
                # approximate q by qT
                nf = P.nfs[k]
                qT = q(Paux.xT,i)
                out.q[i] += 0.5 * z(qT,nf.τ,nf.δ,nf.λ) * K̄(qT-nf.δ,nf.τ) * nf.λ
                out.p[i] += 0.5 * dot(p(x,i),nf.λ) * ( z(qT,nf.τ,nf.δ,nf.λ) * ∇K̄(qT - nf.δ,nf.τ) -K̄(qT - nf.δ,nf.τ) * ∇z(qT,nf.τ,nf.δ,nf.λ) )
            end
        end
    end
    out
end



"""
Compute tildeB(t) for landmarks auxiliary process
"""
function Bridge.B(t, Paux::LandmarksAux)
    X = zeros(UncF, 2Paux.n, 2Paux.n)
    for i in 1:Paux.n
        for j in 1:Paux.n
            X[2i-1,2j] =  kernel(q(Paux.xT,i) - q(Paux.xT,j), P) * one(UncF)
        end
        if itostrat
            for k in 1:length(Paux.nfs)
                nf = P.nfs[k]
                qT = q(Paux.xT,i)
                X[2i,2i] += 0.5 * ( z(qT,nf.τ,nf.δ,nf.λ) * ∇K̄(qT - nf.δ,nf.τ) -K̄(qT - nf.δ,nf.τ) * ∇z(qT,nf.τ,nf.δ,nf.λ) )  * nf.λ'
            end
        end
    end
    X
end

"""
Compute B̃(t) * X (B̃ from auxiliary process) and write to out
Both B̃(t) and X are of type UncMat
"""
function Bridge.B!(t,X,out, Paux::LandmarksAux)
    out .= 0.0 * out
    u = zero(UncF)
    for i in 1:Paux.n  # separately loop over even and odd indices
        for k in 1:2Paux.n # loop over all columns
            for j in 1:Paux.n
                 out[2i-1,k] += kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)*X[p(j), k]
            end
            if itostrat
                u = 0.0*u
                for k in 1:length(Paux.nfs)
                    nf = P.nfs[k]
                    qT = q(Paux.xT,i)
                    u += 0.5 * ( z(qT,nf.τ,nf.δ,nf.λ) * ∇K̄(qT - nf.δ,nf.τ) -K̄(qT - nf.δ,nf.τ) * ∇z(qT,nf.τ,nf.δ,nf.λ) )  * nf.λ'
                end
                out[2i,k] = u * X[2i,k]
            end
        end
    end
    out
end

function Bridge.β(t, Paux::LandmarksAux)
    out = 0.0 * copy(Paux.xT.q)
    if itostrat
        for i in 1:Paux.n
            for k in 1:length(P.nfs)
                nf = P.nfs[k]
                qT = q(Paux.xT,i)
                out[i] += 0.5 * z(qT,nf.τ,nf.δ,nf.λ) * K̄(qT-nf.δ,nf.τ) * nf.λ # simply take q at endpoint
            end
        end
        return(State(out,zeros(PointF,Paux.n)))
    else
        return (State(zeros(PointF,Paux.n), zeros(PointF,Paux.n)) )
    end
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
function σtmul(t, x_, y::State{Pnt}, P::Union{Landmarks,LandmarksAux}) where Pnt
    if P isa Landmarks
        x = x_
    else
        x = P.xT
    end
    out = zeros(Pnt, length(P.nfs))
    for j in 1:length(P.nfs)
        for i in 1:P.n
            out[j] += σq(q(x, i), P.nfs[j])' * y.q[i] +
                        σp(q(x, i), p(x, i), P.nfs[j])' * y.p[i]
        end
    end
    out
end

function σt!(t, x_, y::State{Pnt}, out, P::Union{Landmarks,LandmarksAux}) where Pnt
    zero!(out)
    if P isa Landmarks
        x = x_
    else
        x = P.xT
    end
    for j in 1:length(P.nfs)
        for i in 1:P.n
            out[j] += σq(q(x, i), P.nfs[j])' * q(y, i) +
                    σp(q(x, i), p(x, i), P.nfs[j])' * p(y, i)
        end
    end
    out
end

function Bridge.a(t, x_, P::Union{Landmarks,LandmarksAux})
    if P isa Landmarks
        x = x_
    else
        x = P.xT
    end
    out = zeros(Unc{deepeltype(x)}, 2P.n,2P.n)
    for i in 1:P.n
        for k in i:P.n
            for j in 1:length(P.nfs)
                a11 = σq(q(x,i),P.nfs[j])
                a21 = σp(q(x,i),p(x,i),P.nfs[j])
                a12 = σq(q(x,k),P.nfs[j])
                a22 = σp(q(x,k),p(x,k),P.nfs[j])
                out[2i-1,2k-1] += a11 * a12'
                out[2i-1,2k] += a11 * a22'
                out[2i,2k-1] += a21 * a12'
                out[2i,2k] += a21 * a22'
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


"""
Multiply a(t,x) times a vector of points
Returns a State
(first multiply with sigma', via function σtmul, next left-multiply this vector with σ)
"""
function amul(t, x::State, xin::Vector{<:Point}, P::Union{Landmarks,LandmarksAux})
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

function Bridge.a!(t, x_, out, P::Union{Landmarks,LandmarksAux})
    zero!(out)
    if P isa Landmarks
        x = x_
    else
        x = P.xT
    end
    for i in 1:P.n
        for k in i:P.n
            for j in 1:length(P.nfs)
                a11 = σq(q(x,i),P.nfs[j])
                a21 = σp(q(x,i),p(x,i),P.nfs[j])
                a12 = σq(q(x,k),P.nfs[j])
                a22 = σp(q(x,k),p(x,k),P.nfs[j])
                out[2i-1,2k-1] += a11 * a12'
                out[2i-1,2k] += a11 * a22'
                out[2i,2k-1] += a21 * a12'
                out[2i,2k] += a21 * a22'
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


######################################################################################################


if TEST
    Hend⁺ = [rand(UncF) for i in 1:2Paux.n, j in 1:2Paux.n]
    t0 = 2.0
    BB = Bridge.B(t0,Paux) * Hend⁺
    out = deepcopy(Hend⁺)
    Bridge.B!(t0,Hend⁺,out,Paux)
    @test out==BB
end





if false # some failed attempts to compute tr(aH) right away faster, with less memory allocation

    """
    Compute row_i{sigmaq(t,x)} * dm where dm is a vector of points and sigma is the diffusion coefficient of landmarks
    write to out which is of Vector{Points}
    """
    function σq!(t, x_, dm, out, P::Union{Landmarks,LandmarksAux})
        if P isa Landmarks
            x = x_
        else
            x = P.xT
        end
        zero!(out)
        for j in 1:length(P.nfs)
            out += σq(q(x, i), P.nfs[j]) * dm[j]
        end

        out
    end


    """
    Compute row_i{sigmap(t,x)} * dm where dm is a vector of points and sigma is the diffusion coefficient of landmarks
    write to out which is of Vector{Points}
    """
    function σp!(t, x_, dm, out, P::Union{Landmarks,LandmarksAux})
        if P isa Landmarks
            x = x_
        else
            x = P.xT
        end
        zero!(out)
        for j in 1:length(P.nfs)
            out += σp(q(x, i), p(x, i), P.nfs[j]) * dm[j]
        end
        out
    end
    #######################
    function tr_aH(t, x::State,H, P::Union{Landmarks,LandmarksAux})
        som = 0.0
        outq = zeros(UncF, length(P.nfs))
        outp = zeros(UncF, length(P.nfs))
        for i in 1:P.n
            σtHcol!(outq,x,H[:,2i-1],P.nfs) # compute σ' times Col_i(H) (should give J x 1 column-vector of Unc)
            σtHcol!(outp,x,H[:,2i],P.nfs) # compute σ' times Col_i(H) (should give J x 1 column-vector of Unc)
            for j in 1:length(P.nfs)
                    som += tr(σq(q(x,i),P.nfs[j]) * outq[j]) +
                                tr(σp(q(x,i),p(x,i),P.nfs[j]) * outp[j])
            end

        end
        som
    end

    function σtHcol!(out,x,Hvec,nfs)
        zero!(out)
        for j in 1:length(nfs)
            for i in 1:P.n
                out[j] += σq(q(x,i),nfs[j]) * Hvec[2i-1] +  σp(q(x,i),p(x,i),nfs[j]) * Hvec[2i]
            end
        end
        out
    end

    # @time tr_aH(1.0,x0,H,P)
    # @time  sum(sum(Bridge.a((2,3.0,), x0, P).*H))
    # @time dot(Bridge.a((2,3.0,), x0, P),H)
end
