function partialbridgeodeνH!(::R3, t, L, Σ, v, νt, Ht, P, ϵ)
    m, d = size(L)
    # print(typeof(H⁺t))
    # print(typeof(νt))
    # print(typeof(inv(L' * inv(Σ) * L + ϵ * I)   ))
    Ht[end] = (L' * inv(Σ) * L + ϵ * I)
    H⁺ = inv(Ht[end])
    νt[end] = H⁺ * L' * inv(Σ) * v
    ν = νt[end]
    F(t, y, P) = B(t, P)*y + β(t,P)
    Ri(t, y, P) = B(t, P)*y + y * B(t,P)' - a(t, P)
    for i in length(t)-1:-1:1
        dt = t[i] - t[i+1]
        ν = kernelr3(F, t[i+1], ν, dt, P)
        H⁺ = kernelr3(Ri, t[i+1], H⁺, dt, P)

        νt[i] = ν
        Ht[i] = inv(H⁺)
    end
    νt, Ht
 end

struct Lyap
end
function partialbridgeodeνH!(::Lyap, t, νt, Ht, P, νend, Hend⁺)
    Ht[end] = inv(Hend⁺)
    νt[end] = νend
    H⁺ = Hend⁺
    ν = νend
    F(t, y, P) = B(t, P)*y + β(t,P)

    for i in length(t)-1:-1:1
        dt = t[i] - t[i+1]
        ν = kernelr3(F, t[i+1], ν, dt, P)
        H⁺ = lyapunovpsdbackward_step(t[i+1], H⁺, -dt, P)
        νt[i] = ν
        Ht[i] = inv(H⁺)
    end
    ν, H⁺
end


"""
    PartialBridgeνH

Guided proposal process for diffusion bridge using backward recursion.

    PartialBridgeνH(tt, P, Pt, L, v,ϵ Σ)

    Guided proposal process for a partial diffusion bridge of `P` to `v` on
    the time grid `tt` using guiding term derived from linear process `Pt`.

    PartialBridgeνH(tt, P, Pt, ν, Hend⁺)

    Guided proposal process on the time grid `tt` using guiding term derived from
    linear process `Pt` with backwards equation initialized at `ν, Hend⁺`.

"""
struct PartialBridgeνH{T,TP,TPt,Tν,TH} <: ContinuousTimeProcess{T}
    Target::TP
    Pt::TPt
    tt::Vector{Float64}
    ν::Vector{Tν}
    H::Vector{TH}
    PartialBridgeνH(P::TP, Pt::TPt, tt, νt::Vector{Tν}, Ht::Vector{TH}) where {TP,TPt,Tν,TH} =
        new{Bridge.valtype(P),TP,TPt,Tν,TH}(P, Pt, tt, νt, Ht)


    # 6-7 arg
    function PartialBridgeνH(tt_, P, Pt, L, v::Tv, ϵ, Σ = Bridge.outer(zero(v))) where {Tv}
        tt = collect(tt_)
        N = length(tt)
        m, d = size(L)
        TH = typeof(SMatrix{d,d}(1.0I))
        Ht = zeros(TH, N)
        Tν = typeof(@SVector zeros(d))
        νt = zeros(Tν, N)
        partialbridgeodeνH!(R3(), tt, L, Σ, v, νt, Ht, Pt, ϵ)
        PartialBridgeνH(P, Pt, tt, νt, Ht)
    end
end
# 5 arg
function partialbridgeνH(tt_, P, Pt, νend::Tv, Hend⁺::TH) where {Tv,TH}
    tt = collect(tt_)
    N = length(tt)
    Ht = zeros(TH, N)
    νt = zeros(Tv, N)
    ν, H⁺ = partialbridgeodeνH!(Lyap(), tt, νt, Ht, Pt, νend, Hend⁺)
    PartialBridgeνH(P, Pt, tt, νt, Ht), ν, H⁺
end

function _b((i,t)::IndexedTime, x, P::PartialBridgeνH)
    b(t, x, P.Target) + a(t, x, P.Target)*(P.H[i]*(P.ν[i] - x))
end

r((i,t)::IndexedTime, x, P::PartialBridgeνH) = P.H[i]*(P.ν[i] - x)
H((i,t)::IndexedTime, x, P::PartialBridgeνH) = P.H[i]

σ(t, x, P::PartialBridgeνH) = σ(t, x, P.Target)
a(t, x, P::PartialBridgeνH) = a(t, x, P.Target)
Γ(t, x, P::PartialBridgeνH) = Γ(t, x, P.Target)
constdiff(P::PartialBridgeνH) = constdiff(P.Target) && constdiff(P.Pt)



function llikelihood(::LeftRule, Xcirc::SamplePath, Po::PartialBridgeνH; skip = 0)
    tt = Xcirc.tt
    xx = Xcirc.yy

    som::Float64 = 0.
    for i in 1:length(tt)-1-skip #skip last value, summing over n-1 elements
        s = tt[i]
        x = xx[i]
        r = Bridge.r((i,s), x, Po)

        som += dot( _b((i,s), x, target(Po)) - _b((i,s), x, auxiliary(Po)), r ) * (tt[i+1]-tt[i])
        if !constdiff(Po)
            H = H((i,s), x, Po)
            som -= 0.5*tr( (a((i,s), x, target(Po)) - atilde((i,s), x, Po))*(H) ) * (tt[i+1]-tt[i])
            som += 0.5*( r'*(a((i,s), x, target(Po)) - atilde((i,s), x, Po))*r ) * (tt[i+1]-tt[i])
        end
    end
    som
end
