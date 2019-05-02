# compute guiding term: backward ode
# compute likelihood of guided proposal
using LinearAlgebra

import Bridge: kernelr3!, R3!, target, auxiliary, constdiff, llikelihood, _b!, B!, σ!, b!, σ, b


function gpupdate(ν, P, Σ, L, v)
    if all(diag(P) .== Inf)
        P_ = inv(L' * inv(Σ) * L)
        V_ = (L' * inv(Σ) * L)\(L' * inv(Σ) *  v)
        return V_, P_
    else
        Z = I - P*L'*inv(Σ + L*P*L')*L
        return Z*P*L'*inv(Σ)*v + Z*ν, Z*P
    end
end

function gpupdate(ν, P, C, Σ, L, v)
    m, d = size(L)
    @assert m == length(v)
    if all(diag(P) .== Inf)
        P_ = inv(L' * inv(Σ) * L)
        V_ = (L' * inv(Σ) * L)\(L' * inv(Σ) *  v)
        C += 0.5 * dot(v, Σ\v)
        C += length(v)/2*log(2π) + 0.5*logdet(Σ)
        return V_, P_, C
    else
        Z = I - P*L'*inv(Σ + L*P*L')*L
        C += 0.5 * dot(v, Σ\v)
        C += length(v)/2*log(2π) + 0.5*logdet(Σ)
        return Z*P*L'*inv(Σ)*v + Z*ν, Z*P, C

    end
end


"""
(ν, P) are the values just to the right of time T,
(Σ, L, v) are the noise covariance, observations matrix L and observation at time T
"""
function gpupdate(ν2::State, P2, C, Σ, L, v2::Vector{<:Point})

    ν = deepvec(ν2)
    P = deepmat(P2)
    v = reinterpret(Float64,v2)
    if all(diag(P) .== Inf)
        P_ = inv(L' * inv(Σ) * L)
        Σᴵv = inv(Σ) * v
        V_ = (L' * inv(Σ) * L)\(L' * Σᴵv)
        # P_ = inv(L' * inv(Σ) * L + 10^(-6)*I)
        # V_ = P_ * L' * inv(Σ) * v
        C += 0.5 * dot(v, Σᴵv)
        C += length(v)/2*log(2π) + 0.5*logdet(Σ)
        return deepvec2state(V_), deepmat2unc(P_), C
    else
        Z = I - P*L'*inv(Σ + L*P*L')*L
        C += 0.5 * dot(v, Σ\v)
        C += length(v)/2*log(2π) + 0.5*logdet(Σ)

        return deepvec2state(Z*P*L'*inv(Σ)*v + Z*ν), deepmat2unc(Z*P), C
         # P_ = inv(L' * inv(Σ) * L + inv(P))
         # V_ = P_ * L' * inv(Σ) * v + P_ * inv(P) * ν
         # return deepvec2state(V_), deepmat2unc(P_)
    end
end

struct Arg4Closure{B,T}
    f::B
    arg4::T
end
(f::Arg4Closure)(arg1, arg2, arg3) = f.f(arg1, arg2, arg3, f.arg4)

"""
Replacing out with dP, which is  B*arg + arg*B'- tilde_a
"""
function dP!(t, p, out, P::Union{LandmarksAux, MarslandShardlowAux})
    B!(t, p, out, P)
    out .= out .+ out' - a(t, P)
    out
end

# """
# Make a 4-tuple where each tuple contains a copy of y
# """
# Bridge.workspace(::Bridge.R3!, y) = (copy(y), copy(y), copy(y), copy(y))

"""
Computes solutions to backwards filtering odes for ν and H⁺ and an interval, say (S,T].
Initialisations for the odes on the right are given by arguments ν, H⁺, C
Writes ν and H⁺ into νt and H⁺t (for all elements in t) and returns ν, H⁺, H, C at S+
"""
function bucybackwards!(S::R3!, t, νt, H⁺t, Ht, Paux, ν, H⁺, C)
    H⁺t[end] = H⁺
    Ht[end] = InverseCholesky(lchol(H⁺))
    νt[end] = ν
    wsν = Bridge.workspace(S, ν)
    wsH⁺ = Bridge.workspace(S, H⁺)
    b! = Arg4Closure(Bridge.b!, Paux)
    dP! = Arg4Closure(Bridge.dP!, Paux)
    for i in length(t)-1:-1:1
        dt = t[i] - t[i+1]
        kernelr3!(b!, t[i+1], νt[i+1], wsν, νt[i], dt)
        kernelr3!(dP!, t[i+1], H⁺t[i+1], wsH⁺, H⁺t[i], dt)
        Ht[i] = InverseCholesky(lchol(H⁺t[i]))
        F = Ht[i+1]*νt[i+1]
        C += dot(Bridge.β(t, Paux),F)*dt
        C += 0.5*dot(F, Bridge.a(t, Paux)*F)*dt
        C += -0.5*tr(tr(Ht[i+1]*Matrix(Bridge.a(t, Paux))))*dt
        # FIXME converting to full
    end
    νt[1], H⁺t[1], Ht[1], C
end


struct LRR end  # identifier to call Bucy backwards for low rank riccati

function bucybackwards!(scheme::LRR, t, νt, (St, Ut), Paux, νend, (Send, Uend))
    St[end], Ut[end] = Send, Uend
    νt[end] = νend
    wsν = Bridge.workspace(R3!(), νend)
    ã = deepmat(Bridge.a(t[end],Paux))
    B̃ = sparse(deepmat(Bridge.B(t[end],Paux)))
    for i in length(t)-1:-1:1
        dt = t[i] - t[i+1]
        kernelr3!(Arg4Closure(Bridge.b!, Paux), t[i+1], νt[i+1], wsν, νt[i], dt)
        LowrankRiccati.lowrankriccati!(t[i], t[i+1], -B̃, ã , (St[i+1], Ut[i+1]), (St[i], Ut[i]))

    end
    νt[1], (St[1], Ut[1])
end


### for lyanpunov psd still needs to be fixed
struct Lyap  end

function bucybackwards!(::Lyap, t, νt, H⁺t, Paux, νend, Hend⁺)
    H⁺t[end] = Hend⁺
    νt[end] = νend
    wsν = Bridge.workspace(R3!(), νend)
    B̃ = Matrix(Bridge.B(0.0, Paux))
    ã = Bridge.a(0.0, Paux)
    for i in length(t)-1:-1:1
        dt = t[i] - t[i+1]
        kernelr3!(Arg4Closure(Bridge.b!, Paux), t[i+1], νt[i+1], wsν, νt[i], dt)
        lyapunovpsdbackward_step!(t[i+1], -dt, Paux, H⁺t[i+1], H⁺t[i], B̃, ã) # maybe with -dt
    end
    νt[1], H⁺t[1]
end


"""
Construct guided proposal on a single segment with times in tt from precomputed ν and H
"""
struct GuidedProposal!{T,Ttarget,Taux,Tν,TH,TC,F} <: ContinuousTimeProcess{T}
    target::Ttarget   # P
    aux::Taux      # Ptilde
    tt::Vector{Float64}  # grid of time points on single segment (S,T]
    ν::Vector{Tν}
    H::Vector{TH}
    C::TC
    endpoint::F

    function GuidedProposal!(target, aux, tt_, ν, H, C, endpoint=Bridge.endpoint)
        tt = collect(tt_)
        new{Bridge.valtype(target),typeof(target),typeof(aux),eltype(ν),eltype(H),typeof(C),typeof(endpoint)}(target, aux, tt, ν, H, C, endpoint)
    end
end

Bridge.lptilde(x, Po::GuidedProposal!) = -0.5*(dot(x, Po.H[1]*x) - 2dot(x, Po.H[1]*Po.ν[1])) - Po.C


function Bridge._b!((i,t), x, out, P::GuidedProposal!)
    Bridge.b!(t, x, out, P.target)
    out .+= amul(t,x,P.H[i]*(P.ν[i] - x),P.target)
    out
end

function Bridge._b((i,t), x, P::GuidedProposal!)
    out = Bridge.b(t, x, P.target)
    out .+= amul(t,x,P.H[i]*(P.ν[i] - x),P.target)
    out
end

Bridge.σ!(t, x, dw, out, P::GuidedProposal!) = σ!(t, x, dw, out, P.target)

Bridge.σ(t, x, dw, P::GuidedProposal!) = σ(t, x, dw, P.target)


function _r!((i,t), x, out, P::GuidedProposal!)
    out .= (P.H[i]*(P.ν[i] - x))
    out
end



#H((i,t), x, P::GuidedProposal!) = P.H[i]

target(P::GuidedProposal!) = P.target
auxiliary(P::GuidedProposal!) = P.aux

constdiff(P::GuidedProposal!) = constdiff(target(P)) && constdiff(auxiliary(P))

function llikelihood(::LeftRule, Xcirc::SamplePath{State{Pnt}}, Q::GuidedProposal!; skip = 0) where {Pnt}
    tt = Xcirc.tt
    xx = Xcirc.yy

    som::eltype(xx[1]) = 0.
    rout = copy(xx[1])

    if !constdiff(Q) # use sizetypes?
        srout = zeros(Pnt, length(P.nfs))
        strout = zeros(Pnt, length(P.nfs))
    end


    bout = copy(rout)
    btout = copy(rout)
    for i in 1:length(tt)-1-skip #skip last value, summing over n-1 elements
        s = tt[i]
        x = xx[i]
        _r!((i,s), x, rout, Q)
        b!(s, x, bout, target(Q))
        _b!((i,s), x, btout, auxiliary(Q))

        dt = tt[i+1]-tt[i]
        #dump(som)
        som += dot(bout-btout, rout) * dt

        if !constdiff(Q)
            #Δa =  a((i,s), x, target(Q)) - a((i,s), x, auxiliary(Q))
            # H = H((i,s), x, auxiliary(Q))
            # som -= 0.5*tr(Δa*H) * dt
            # som += 0.5*(rout'*Δa*rout) * dt

            σt!(s, x, rout, srout, target(Q))
            σt!(s, x, rout, strout, auxiliary(Q))
            error("NOT IMPLEMENTED YET")
            #Δa = Bridge.a(s, x, target(Q)) - Bridge.a(s,  auxiliary(Q))
            #H = Bridge.H((i,s), x, auxiliary(Q))
            # som -= 0.5*tr(Δa*H) * dt
             #som += 0.5*dot(rout,Δa*rout) * dt
             som += 0.5*Bridge.inner(srout) * dt
             som -= 0.5*Bridge.inner(strout) * dt
        end
    end
    som
end
