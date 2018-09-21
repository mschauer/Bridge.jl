
function partialbridgeode!(::R3, t, L, Σ, Lt, Mt, μt, P)
    m, d = size(L)
    Lt[end] = L
    Mt[end] = inv(Σ)
    M⁺ = Σ
    μt[end] = μ = 0*L[:,1]

    @assert size(L[:,1]) == (m,)
    @assert size(L*L') == size(Σ) == (m, m)

    for i in length(t)-1:-1:1
        dt = t[i] - t[i+1]
        L = kernelr3((t, y, P) -> -y*B(t, P), t[i+1], L, dt, P)
        M⁺ = kernelr3((t, y, (L,P)) -> -outer(L*σ(t, P)), t[i+1], M⁺, dt, (L,P))
        μ = kernelr3((t, y, (L,P)) -> -L*β(t, P), t[i+1], μ, dt, (L,P))

        Lt[i] = L
        Mt[i] = inv(M⁺)
        μt[i] = μ
    end
    Lt, Mt, μt
 end


"""
    PartialBridge

Guided proposal process for diffusion bridge using backward recursion.

    PartialBridge(tt, P, Pt,  L, v, Σ)

Guided proposal process for a partial diffusion bridge of `P` to `v` on
the time grid `tt` using guiding term derived from linear process `Pt`.

Simulate with `bridge!`.
"""

struct PartialBridge{T,R,R2,Tv,TL,TM} <: ContinuousTimeProcess{T}
    Target::R
    Pt::R2
    tt::Vector{Float64}
    v::Tv
    L::Vector{TL}
    M::Vector{TM}
    μ::Vector{Tv}

    function PartialBridge(tt_, P, Pt, L::TL, v::Tv, Σ::TM = Bridge.outer(zero(v))) where {TL, Tv, TM}
        tt = collect(tt_)
        N = length(tt)
        Lt = zeros(TL, N)
        Mt = zeros(TM, N)
        μt = zeros(Tv, N)
        partialbridgeode!(R3(), tt, L, Σ, Lt, Mt, μt, Pt)
        new{Bridge.valtype(P),typeof(P),typeof(Pt),Tv,TL,TM}(P, Pt, tt, v, Lt, Mt, μt)
    end
end

function bi(i::Integer, x, P::PartialBridge)
    b(P.tt[i], x, P.Target) + a(P.tt[i], x, P.Target)*P.L[i]'*P.M[i]*(P.v - P.μ[i] -  P.L[i]*x)
end

ri(i::Integer, x, P::PartialBridge) = P.L[i]'*P.M[i]*(P.v - P.μ[i] -  P.L[i]*x)
Hi(i::Integer, x, P::PartialBridge) = P.L[i]' * P.M[i] * P.L[i]


σ(t, x, P::PartialBridge) = σ(t, x, P.Target)
a(t, x, P::PartialBridge) = a(t, x, P.Target)
Γ(t, x, P::PartialBridge) = Γ(t, x, P.Target)
constdiff(P::PartialBridge) = constdiff(P.Target) && constdiff(P.Pt)
btilde(t, x, P::PartialBridge) = b(t, x, P.Pt)
atilde(t, x, P::PartialBridge) = a(t, x, P.Pt)
aitilde(t, x, P::PartialBridge) = ai(t, x, P.Pt)
bitilde(i, x, P::PartialBridge) = bi(i, x, P.Pt)

hasbi(::PartialBridge) = true
hasbitilde(P::PartialBridge) = hasbi(P.Pt)
hasaitilde(P::PartialBridge) = hasai(P.Pt)




function llikelihood(::LeftRule, Xcirc::SamplePath, Po::PartialBridge; skip = 0)
    tt = Xcirc.tt
    xx = Xcirc.yy

    som::Float64 = 0.
    for i in 1:length(tt)-1-skip #skip last value, summing over n-1 elements
        s = tt[i]
        x = xx[i]
        r = Bridge.ri(i, x, Po)

        if hasbitilde(Po)
            som += dot( b(s, x, Po.Target) - bitilde(i, x, Po), r ) * (tt[i+1]-tt[i])
        else
            som += dot( b(s, x, Po.Target) - btilde(s, x, Po), r ) * (tt[i+1]-tt[i])
        end
        if !constdiff(Po)
            H = Hi(i, x, Po)
            if hasaitilde(Po)
                som -= 0.5*tr( (a(s, x, Po.Target) - aitilde(i, x, Po))*(H) ) * (tt[i+1]-tt[i])
                som += 0.5*( r'*(a(s, x, Po.Target) - aitilde(i, x, Po))*r ) * (tt[i+1]-tt[i])
            else
                som -= 0.5*tr( (a(s, x, Po.Target) - atilde(s, x, Po))*(H) ) * (tt[i+1]-tt[i])
                som += 0.5*( r'*(a(s, x, Po.Target) - atilde(s, x, Po))*r ) * (tt[i+1]-tt[i])
            end
        end
    end
    som
end

