struct Arg4Closure{B,T}
    f::B
    arg4::T
end
(f::Arg4Closure)(arg1, arg2, arg3) = f.f(arg1, arg2, arg3, f.arg4)

function dP!(t, p, out, P)
    B!(t, p, out, P)
    out .= out .+ out' - a(t, P)
    out
end

function partialbridgeode!(S::R3!, t, L, Σnoise, v, νt, Σt, P, ϵ)
    m, d = size(L)
    # print(typeof(Σt))
    # print(typeof(νt))
    # print(typeof(inv(L' * inv(Σ) * L + ϵ * I)   ))
    Γnoise = iszero(Σnoise) ? inv(eps()*one(Σnoise)) : inv(Σnoise)
    Σt[end] .=  inv(L' * Γnoise * L + ϵ * I)
    νt[end] .=  Σt[end] * L' * Γnoise * v
    wsν = workspace(S, νt[end])
    wsΣ = workspace(S, Σt[end])

    for i in length(t)-1:-1:1
        dt = t[i] - t[i+1]
        kernelr3!(Arg4Closure(b!, P), t[i+1], νt[i+1], wsν, νt[i], dt)
        kernelr3!(Arg4Closure(dP!, P), t[i+1], Σt[i+1], wsΣ, Σt[i], dt)
    end
    νt, Σt
 end

struct PartialBridge!{T,TP,TPt,Tv,Tν,TH} <: ContinuousTimeProcess{T}
    Target::TP
    Pt::TPt
    tt::Vector{Float64}
    v::Tv
    ν::Vector{Tν}
    H::Vector{TH}

    function PartialBridge!(tt_, P, Pt, L, v::Tv, ϵ, Σnoise = Bridge.outer(zero(v))) where {Tv}
        tt = collect(tt_)
        N = length(tt)
        m, d = size(L)
        Σ = L'*L
        ν = diag(Σ)
        Σt = [zero(outer(ν)) for i in 1:N]
        νt = [zero(ν) for i in 1:N]

        TH = typeof(Σ)
        Tν = typeof(ν)

        partialbridgeode!(R3!(), tt, L, Σnoise, v, νt, Σt, Pt,ϵ)
        map!(inv, Σt, Σt)
        new{Bridge.valtype(P),typeof(P),typeof(Pt),Tv,Tν,TH}(P, Pt, tt, v, νt, Σt)
    end
end


function _b!((i,t), x, out, P::PartialBridge!)
    b!(t, x, out, P.Target)
    out .+= a(t, x, P.Target)*(P.H[i]*(P.ν[i] - x))
    out
end
σ!(t, x, dw, out, P::PartialBridge!) = σ!(t, x, dw, out, P.Target)


function rti!((t,i), x, out, P::PartialBridge!)
    out .= (P.H[i]*(P.ν[i] - x))
    out
end

btitilde!((t,i), x, out, P) = _b!((t,i), x, out, P.Pt)

constdiff(P::PartialBridge!) = constdiff(P.Target) && constdiff(P.Pt)


function llikelihood(::LeftRule, Xcirc::SamplePath, Po::PartialBridge!; skip = 0)
    tt = Xcirc.tt
    xx = Xcirc.yy

    som::Float64 = 0.
    rout = copy(xx[1])
    bout = copy(rout)
    btout = copy(rout)
    for i in 1:length(tt)-1-skip #skip last value, summing over n-1 elements
        s = tt[i]
        x = xx[i]
        rti!((s,i), x, rout, Po)
        b!(s, x, bout, target(Po))
        btitilde!((s,i), x, btout, Po)

        som += dot(bout, rout) * (tt[i+1]-tt[i])
        som -= dot(btout, rout) * (tt[i+1]-tt[i])

        if !constdiff(Po)
            error("not implemented")
        end
    end
    som
end
