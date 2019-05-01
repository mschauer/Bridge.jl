function updateνH⁺(L, Σ, v, ϵ)
    m, d = size(L)
    @assert m == length(v)
    H = (L' * inv(Σ) * L + ϵ * I)
    H⁺ = inv(H)
    ν = H⁺ * L' * inv(Σ) * v
    ν, H⁺
end
function updateC(L, Σ, v, C = 0.0)
    m, d = size(L)
    C += 0.5 * dot(v, Σ\v)
    C += length(m)/2*log(2pi) + 0.5*logdet(Σ)
    C
end
function updateνH⁺C(L, Σ, v, ϵ)
    updateνH⁺(L, Σ, v, ϵ)..., updateC(L, Σ, v)
end


function partialbridgeodeνH!(::R3, t, νt, Ht, P, (ν, H⁺, C))
    #m, d = size(L)
    #@assert m == length(v)
    # print(typeof(H⁺t))
    # print(typeof(νt))
    # print(typeof(inv(L' * inv(Σ) * L + ϵ * I)   ))
    #Ht[end] = H = (L' * inv(Σ) * L + ϵ * I)
    #H⁺ = inv(H)
    #νt[end] = H⁺ * L' * inv(Σ) * v
    #F = L' * inv(Σ) * v
    #ν = νt[end]
    #C += 0.5 * v' * inv(Σ) * v # 0.5 c
    #C += m/2*log(2pi) + 0.5*log(abs(det(Σ))) # Q, use logabsdet, actually
    #ν, H⁺ = zero(νt[end]), zero(Ht[end])
    #H⁺, ν = gpupdate(H⁺, ν, L, Σ, v)
    H = inv(H⁺)
    #C = updateC(L, Σ, v, C)

    b̃(t, y, P) = B(t, P)*y + β(t, P)
    dH⁺(t, y, P) = B(t, P)*y + y * B(t,P)' - a(t, P)
#    dH(t, y, P) = - B(t, P)'*y - y * B(t,P) + y*a(t, P)*y'
#    dF(t, y, (H,P)) = -B(t, P)'*y + H*a(t, P)*y  + H*β(t, P)

    for i in length(t)-1:-1:1
        dt = t[i] - t[i+1]
        ν = kernelr3(b̃, t[i+1], ν, dt, P)
        H⁺ = kernelr3(dH⁺, t[i+1], H⁺, dt, P)
        #H = kernelr3(dH, t[i+1], H, dt, P)



        #F = kernelr3(dF, t[i+1], F, dt, (H,P))
        F = Ht[i+1]*νt[i+1]
        #@show F , Ht[i+1]*νt[i+1]
        C += β(t[i+1], P)'*F*dt  + 0.5*F'*a(t[i+1], P)*F*dt - 0.5*tr(H*a(t[i+1], P))*dt

        νt[i] = ν
        #@show H, inv(H⁺)
        Ht[i] = H = inv(H⁺)
    end

    νt, Ht, C
end

function updateFHC(L, Σ, v, F, H, ϵ = 0.0, C = 0.0)
    m, d = size(L)
    H += (L' * inv(Σ) * L + ϵ * I)
    F += L' * inv(Σ) * v
    F, H, updateC(L, Σ, v, C)
end

function partialbridgeodeHνH!(::R3, t, Ft, Ht, P, (F, H, C))
    Ht[end] = H
    Ft[end] = F

    dH(t, y, P) = - B(t, P)'*y - y * B(t,P) + y*a(t, P)*y'
    dF(t, y, (H,P)) = -B(t, P)'*y + H*a(t, P)*y  + H*β(t, P)

    for i in length(t)-1:-1:1
        dt = t[i] - t[i+1]
        C += β(t[i+1], P)'*F*dt + 0.5*F'*a(t[i+1], P)*F*dt - 0.5*tr(H*a(t[i+1], P))*dt
        H = kernelr3(dH, t[i+1], H, dt, P)
        F = kernelr3(dF, t[i+1], F, dt, (H, P))
        Ft[i] = F
        Ht[i] = H
    end

    Ft, Ht, C
end


struct Lyap
end
function partialbridgeodeνH!(::Lyap, t, νt, Ht, P, (νend, Hend⁺, C))
    Ht[end] = H = inv(Hend⁺)
    νt[end] = νend
    H⁺ = Hend⁺
    ν = νend
    b̃(t, y, P) = B(t, P)*y + β(t,P)

    for i in length(t)-1:-1:1
        dt = t[i] - t[i+1]
        ν = kernelr3(b̃, t[i+1], ν, dt, P)
        H⁺ = lyapunovpsdbackward_step(t[i+1], H⁺, -dt, P)
        F = Ht[i+1]*νt[i+1]
        C += β(t[i+1], P)'*F*dt  + 0.5*F'*a(t[i+1], P)*F*dt - 0.5*tr(H*a(t[i+1], P))*dt
        νt[i] = ν
        Ht[i] = H = inv(H⁺)
    end
    ν, H⁺, C
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
struct PartialBridgeνH{T,TP,TPt,Tν,TH,TC} <: ContinuousTimeProcess{T}
    Target::TP
    Pt::TPt
    tt::Vector{Float64}
    ν::Vector{Tν}
    H::Vector{TH}
    C::TC
    PartialBridgeνH(P::TP, Pt::TPt, tt, νt::Vector{Tν}, Ht::Vector{TH}, C::TC=0.0) where {TP,TPt,Tν,TH,TC} =
        new{Bridge.valtype(P),TP,TPt,Tν,TH,TC}(P, Pt, tt, νt, Ht, C)


    # 6-7 arg
    function PartialBridgeνH(tt_, P, Pt, L, v::Tv, ϵ, Σ = Bridge.outer(zero(v))) where {Tv}
        tt = collect(tt_)
        N = length(tt)
        m, d = size(L)
        TH = typeof(SMatrix{d,d}(1.0I))
        Ht = zeros(TH, N)
        Tν = typeof(@SVector zeros(d))
        νt = zeros(Tν, N)
        ν, H⁺, C = updateνH⁺C(L, Σ, v, ϵ)
        _, _, C = partialbridgeodeνH!(R3(), tt, νt, Ht, Pt, (ν, H⁺, C))
        PartialBridgeνH(P, Pt, tt, νt, Ht, C)
    end
end
# 5 arg
function partialbridgeνH(tt_, P, Pt, νend::Tv, Hend⁺::TH) where {Tv,TH}
    tt = collect(tt_)
    N = length(tt)
    Ht = zeros(TH, N)
    νt = zeros(Tv, N)
    ν, H⁺, C = partialbridgeodeνH!(Lyap(), tt, νt, Ht, Pt, (νend, Hend⁺, C))
    PartialBridgeνH(P, Pt, tt, νt, Ht, C), ν, H⁺, C
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

lptilde(x, P::PartialBridgeνH) = -0.5*((P.ν[1] - x)'*P.H[1] * (P.ν - x)) - P.C

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
