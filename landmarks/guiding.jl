# compute guiding term: backward ode
# compute likelihood of guided proposal

import Bridge: kernelr3!, R3!, target, auxiliary, constdiff, llikelihood, _b!, B!, σ!, b!

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


"""
(ν, P) are the values just to the right of time T,
(Σ, L, v) are the noise covariance, observations matrix L and observation at time T
"""
function gpupdate(ν2::State, P2, Σ, L, v2::Vector{Point})
    ν = deepvec(ν2)
    P = deepmat(P2)
    v = reinterpret(Float64,v2)
    if false #all(diag(P) .== Inf)
        #P_ = inv(L' * inv(Σ) * L)
        #V_ = (L' * inv(Σ) * L)\(L' * inv(Σ) *  v)
        P_ = inv(L' * inv(Σ) * L + 10^(-6)*I)
        V_ = P_ * L' * inv(Σ) * v
        return deepvec2state(V_), deepmat2unc(P_)
    else
        Z = I - P*L'*inv(Σ + L*P*L')*L
       return deepvec2state(Z*P*L'*inv(Σ)*v + Z*ν), deepmat2unc(Z*P)

        # P_ = inv(L' * inv(Σ) * L + P)
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
Computes solutions to backwards filtering odes for ν and H⁺ an an interval, say (S,T].
Initialisations for the odes on the right are given by νend and Hend⁺
Writes ν and H⁺ into νt and H⁺t (for all elements in t) and returns ν and H⁺ at S+
"""
function bucybackwards!(S::R3!, t, νt, H⁺t, Paux, νend, Hend⁺)
    H⁺t[end] = Hend⁺
    νt[end] = νend
    wsν = Bridge.workspace(S, νend)
    wsH⁺ = Bridge.workspace(S, Hend⁺)
    for i in length(t)-1:-1:1
        dt = t[i] - t[i+1]
        kernelr3!(Arg4Closure(Bridge.b!, Paux), t[i+1], νt[i+1], wsν, νt[i], dt)
        kernelr3!(Arg4Closure(Bridge.dP!, Paux), t[i+1], H⁺t[i+1], wsH⁺, H⁺t[i], dt)
    end
    νt[1], H⁺t[1]
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
        lowrankriccati!(t[i], t[i+1], -B̃, ã , (St[i+1], Ut[i+1]), (St[i], Ut[i]))
    end
    νt[1], (St[1], Ut[1])
end


### for lyanpunov psd still needs to be fixed
struct Lyap  end

function bucybackwards!(::Lyap, t, νt, H⁺t, Paux, νend, Hend⁺)
    H⁺t[end] = Hend⁺
    νt[end] = νend
    wsν = Bridge.workspace(R3!(), νend)
    for i in length(t)-1:-1:1
        dt = t[i] - t[i+1]
        kernelr3!(Arg4Closure(Bridge.b!, Paux), t[i+1], νt[i+1], wsν, νt[i], dt)
        lyapunovpsdbackward_step!(t[i+1], -dt, Paux, H⁺t[i+1], H⁺t[i]) # maybe with -dt
    end
    νt[1], H⁺t[1]
end


"""
Construct guided proposal on a single segment with times in tt from precomputed ν and H
"""
struct GuidedProposal!{T,Ttarget,Taux,Tν,TH,F} <: ContinuousTimeProcess{T}
    target::Ttarget   # P
    aux::Taux      # Ptilde
    tt::Vector{Float64}  # grid of time points on single segment (S,T]
    ν::Vector{Tν}
    H::Vector{TH}
    endpoint::F

    function GuidedProposal!(target, aux, tt_, ν, H, endpoint=Bridge.endpoint)
        tt = collect(tt_)
        new{Bridge.valtype(target),typeof(target),typeof(aux),eltype(ν),eltype(H),typeof(endpoint)}(target, aux, tt, ν, H, endpoint)
    end
end

function Bridge._b!((i,t), x, out, P::GuidedProposal!)
    Bridge.b!(t, x, out, P.target)
    out .+= amul(t,x,P.H[i]*(P.ν[i] - x),P.target)
    out
end

σ!(t, x, dw, out, P::GuidedProposal!) = σ!(t, x, dw, out, P.target)

function _r!((i,t), x, out, P::GuidedProposal!)
    out .= (P.H[i]*(P.ν[i] - x))
    out
end

#H((i,t), x, P::GuidedProposal!) = P.H[i]

target(P::GuidedProposal!) = P.target
auxiliary(P::GuidedProposal!) = P.aux

constdiff(P::GuidedProposal!) = constdiff(target(P)) && constdiff(auxiliary(P))

function llikelihood(::LeftRule, Xcirc::SamplePath, Q::GuidedProposal!; skip = 0)
    tt = Xcirc.tt
    xx = Xcirc.yy

    som::Float64 = 0.
    rout = copy(xx[1])
    bout = copy(rout)
    btout = copy(rout)
    for i in 1:length(tt)-1-skip #skip last value, summing over n-1 elements
        s = tt[i]
        x = xx[i]
        _r!((i,s), x, rout, Q)
        b!(s, x, bout, target(Q))
        _b!((i,s), x, btout, auxiliary(Q))
#        btitilde!((s,i), x, btout, Q)
        dt = tt[i+1]-tt[i]
        som += dot(bout-btout, rout) * dt

        if !constdiff(Q)
            H = H((i,s), x, Q)
            Δa =  a((i,s), x, target(Q)) - a((i,s), x, auxiliary(Q))
            som -= 0.5*tr(Δa*H) * dt
            som += 0.5*(rout'*Δa*rout) * dt
        end
    end
    som
end
