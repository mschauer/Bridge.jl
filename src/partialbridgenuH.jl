function partialbridgeodeνH!(::R3, t, L, Σ, v, νt, H⁺t, P, ϵ)
    m, d = size(L)
    # print(typeof(H⁺t))
    # print(typeof(νt))
    # print(typeof(inv(L' * inv(Σ) * L + ϵ * I)   ))
    H⁺t[end] =  inv(L' * inv(Σ) * L + ϵ * I)
    νt[end] =  H⁺t[end] * L' * inv(Σ) * v
    H⁺ = H⁺t[end]
    ν = νt[end]

    for i in length(t)-1:-1:1
        dt = t[i] - t[i+1]
        ν = kernelr3((t, y, P) -> B(t, P)*y + β(t,P), t[i+1], ν, dt, P)
        H⁺ = kernelr3((t,  y, (L,P)) -> B(t, P)*y + y * B(t,P)'-a(t, P), t[i+1], H⁺, dt, (L,P))

        νt[i] = ν
        H⁺t[i] = H⁺
    end
    νt, H⁺t
 end


"""
    PartialBridgeνH

Guided proposal process for diffusion bridge using backward recursion.

    PartialBridgeνH(tt, P, Pt,  L, v,ϵ Σ)

Guided proposal process for a partial diffusion bridge of `P` to `v` on
the time grid `tt` using guiding term derived from linear process `Pt`.

Simulate with `bridge!`.
"""
struct PartialBridgeνH{T,TP,TPt,Tv,Tν,TH} <: ContinuousTimeProcess{T}
    Target::TP
    Pt::TPt
    tt::Vector{Float64}
    v::Tv
    ν::Vector{Tν}
    H⁺::Vector{TH}

    function PartialBridgeνH(tt_, P, Pt, L, v::Tv,ϵ, Σ = Bridge.outer(zero(v))) where {Tv}
        tt = collect(tt_)
        N = length(tt)
        m, d = size(L)
        TH = typeof(SMatrix{d,d}(1.0I))
        H⁺t = zeros(TH, N)
        Tν = typeof(@SVector zeros(d))
        νt = zeros(Tν, N)

        partialbridgeodeνH!(R3(), tt, L, Σ, v, νt, H⁺t, Pt,ϵ)
        new{Bridge.valtype(P),typeof(P),typeof(Pt),Tv,Tν,TH}(P, Pt, tt, v, νt, H⁺t)
    end
end


function bi(i::Integer, x, P::PartialBridgeνH)
    b(P.tt[i], x, P.Target) + a(P.tt[i], x, P.Target)*(P.H⁺[i]\(P.ν[i] - x))
end

ri(i::Integer, x, P::PartialBridgeνH) = P.H⁺[i]\(P.ν[i] - x)
Hi(i::Integer, x, P::PartialBridgeνH) = inv(P.H⁺[i])

σ(t, x, P::PartialBridgeνH) = σ(t, x, P.Target)
a(t, x, P::PartialBridgeνH) = a(t, x, P.Target)
Γ(t, x, P::PartialBridgeνH) = Γ(t, x, P.Target)
constdiff(P::PartialBridgeνH) = constdiff(P.Target) && constdiff(P.Pt)
btilde(t, x, P::PartialBridgeνH) = b(t, x, P.Pt)
atilde(t, x, P::PartialBridgeνH) = a(t, x, P.Pt)
aitilde(t, x, P::PartialBridgeνH) = ai(t, x, P.Pt)
bitilde(i, x, P::PartialBridgeνH) = bi(i, x, P.Pt)

hasbi(::PartialBridgeνH) = true
hasbitilde(P::PartialBridgeνH) = hasbi(P.Pt)
hasaitilde(P::PartialBridgeνH) = hasai(P.Pt)




function llikelihood(::LeftRule, Xcirc::SamplePath, Po::PartialBridgeνH; skip = 0)
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
