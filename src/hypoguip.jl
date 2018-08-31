
function hypoode!(::R3, tt, L, Lt, M⁺t, μt, P)
    m, d = size(L)
    Lt[end] = L
    M⁺t[end] = M⁺ = 0*L*L'   
    μt[end] = 0*L[:,1]

    @assert size(L[:,1]) == m
    @assert size(L*L') == (m, m)
    
    for i in length(t)-1:-1:1
        L = kernelr3((t, y, P) -> -y*B(t, P), t[i+1], L, t[i] - t[i+1], P)  
        M⁺ = kernelr3((t, y, (L,P)) -> -L*a(t, P)*L', t[i+1], M⁺, t[i] - t[i+1], (L,P))  
        μ = kernelr3((t, y, (L,P)) -> L*β(t, L), t[i+1], μ, t[i] - t[i+1], (L,P))  
        
        Lt[i] = L
        M⁺t[i] = M⁺
        μt[i] = μ  
    end
    Lt, Mt, μt
 end
 


"""
    HypoBridge

Guided proposal process for diffusion bridge using backward recursion.
    
    HypoBridge(tt, P, Pt, v)

Constructor of guided proposal process for diffusion bridge of `P` to `v` on 
the time grid `tt` using guiding term derived from linear process `Pt`.

    HypoBridge(tt, P, Pt, V, H♢)

Guided proposal process for diffusion bridge of `P` to `v` on 
the time grid `tt` using guiding term derived from linear process `Pt`.
Initialize using [`Bridge.gpupdate`](@ref)(H♢, V, L, Σ, v)
"""   
struct HypoBridge{T,S,S2,R2,R} <: ContinuousTimeProcess{T}
    Target::R
    Pt::R2
    tt::Vector{Float64}
    M♢::Vector{S} 
    L::Vector{S2} 
    V::Vector{T}

    function HypoBridge(tt_, P::R, Pt::R2, v::T, h♢::S = Bridge.outer(zero(v))) where {T,R,R2,S}
        tt = collect(tt_)
        N = length(tt)
        H♢ = SamplePath(tt, zeros(S, N)) 
        V = SamplePath(tt, zeros(T, N)) 
        gpHinv!(H♢, Pt, h♢)
        gpV!(V, Pt, v)
        new{T,S,R2,R}(P, Pt, tt, H♢.yy, V.yy)
    end
    function HypoBridge(tt_, P::R, Pt::Union{LinearNoiseAppr, LinearAppr}, v::T, h♢::S = Bridge.outer(zero(v))) where {T,R,S}
        tt = collect(tt_)
        N = length(tt)
        H♢ = SamplePath(tt, zeros(S, N)) 
        V = SamplePath(tt, zeros(T, N)) 
        solvebackwardi!(R3(), (t, K, iP) ->  Bi(iP...)*K + K*Bi(iP...)' - ai(iP...), H♢, h♢, Pt)
        solvebackwardi!(R3(), (t, v, iP) ->  bi(iP[1], v, iP[2]), V, v, Pt)
        new{T,S,typeof(Pt),R}(P, Pt, tt, H♢.yy, V.yy)
    end
end
 
bi(i::Integer, x, P::HypoBridge) = b(P.tt[i], x, P.Target) + a(P.tt[i], x, P.Target)*(P.H♢[i]\(P.V[i] - x)) 
ri(i::Integer, x, P::HypoBridge) = P.H♢[i]\(P.V[i] - x)
Hi(i::Integer, x, P::HypoBridge) = inv(P.H♢[i])

σ(t, x, P::HypoBridge) = σ(t, x, P.Target)
a(t, x, P::HypoBridge) = a(t, x, P.Target)
Γ(t, x, P::HypoBridge) = Γ(t, x, P.Target)
constdiff(P::HypoBridge) = constdiff(P.Target) && constdiff(P.Pt)
btilde(t, x, P::HypoBridge) = b(t, x, P.Pt)
atilde(t, x, P::HypoBridge) = a(t, x, P.Pt)
aitilde(t, x, P::HypoBridge) = ai(t, x, P.Pt)
bitilde(i, x, P::HypoBridge) = bi(i, x, P.Pt)

@inline _traceB(t, x, P) = tr(Bridge.B(t, P))
traceB(tt, P) = solve(Bridge.R3(), _traceB, tt, 0.0, P)

lptilde(P::HypoBridge, u) = logpdfnormal(P.V[1] - u, P.H♢[1]) - traceB(P.tt, P.Pt)

hasbi(::HypoBridge) = true
hasbitilde(P::HypoBridge) = hasbi(P.Pt)
hasaitilde(P::HypoBridge) = hasai(P.Pt)

# fallback for testing
lptilde2(P::HypoBridge, u) = logpdfnormal(P.V[end] - gpmu(P.tt, u, P.Pt), gpK(P.tt, Bridge.outer(zero(u)), P.Pt))

# H♢_ = H♢ -  H♢*L'*inv(Σ + L*H♢*L')*L*H♢
# V_ = H♢_ * (L'*inv(Σ)*v  + H♢*V)

"""
    gpupdate(H♢, V, L, Σ, v)
    gpupdate(P, L, Σ, v)

Return updated `H♢, V` when observation `v` at time zero with error `Σ` is observed.
"""
function gpupdate(H♢, V, L, Σ, v)
    if all(diag(H♢) .== Inf)
        H♢_ = SMatrix(inv(L' * inv(Σ) * L))
        V_ = (L' * inv(Σ) * L)\(L' * inv(Σ) *  v)
        H♢_, V_
    else
        Z = I - H♢*L'*inv(Σ*I + L*H♢*L')*L
        Z*H♢, Z*H♢*L'*inv(Σ)*v + Z*V
    end
end

function gpupdate(H♢, V::SVector, L, Σ::Float64, v)
    if all(diag(H♢) .== Inf)
        H♢_ = SMatrix(inv(L' * inv(Σ) * L))
        V_ = (L' * inv(Σ) * L)\(L' * inv(Σ) *  v)
        H♢_, V_
    else
        Z = I - H♢*L'*inv(Σ + L*H♢*L')*L
        Z*H♢, SVector(Z*H♢*L'*inv(Σ)*v) + Z*V
    end
end

gpupdate(P::HypoBridge, L, Σ, v) = gpupdate(P.H♢[1], P.V[1], L, Σ, v)

# Alternatives to try
#   KT =  PhiTS*(KS + GP2.H♢[1])*PhiTS'   
#   KT = PhiTS*KS*PhiTS' + KTS 
#   logdet( L*KS*L' + Σ - L*KS*inv(KS + GP2.H♢[1])*KS*L' ) + logdet(KS + GP2.H♢[1]) + 2logdet(PhiTS)
#   logdet( KS + GP2.H♢[1] - KS*L'*inv(L*KS*L' + Σ)*L*KS  ) + logdet(L*KS*L' + Σ ) + 2logdet(PhiTS)
function logdetU(GP1, GP2, L, Σ)
    PhiS = fundamental_matrix(GP1.tt, GP1.Pt)
    PhiTS = fundamental_matrix(GP2.tt, GP2.Pt)
    K =  PhiS*GP1.H♢[1]*PhiS' - GP1.H♢[end]
    H = GP2.H♢[1]
    logdet(inv(K) + L'*inv(Σ)*L + inv(H)) + logdet(Σ) + logdet(H) + logdet(K) + 2logdet(PhiTS)
end
