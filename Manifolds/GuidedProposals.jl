"""
    The function kernelr3 is used to apply the trapezoidal method
"""

function kernelr3(f, t, y, dt, P)
    k1 = f(t, y, P)
    k2 = f(t + 1/2*dt, y + 1/2*dt*k1, P)
    k3 = f(t + 3/4*dt, y + 3/4*dt*k2, P)
    y + dt*(2/9*k1 + 1/3*k2 + 4/9*k3)
end


"""
  The struct GuidedProposal generates guided proposals given an endpoint, a
  target process and an auxiliary process. It returns the functions L(t), M(t)
  and μ(t), found by using the trapezoidal method  to solve
  the differential equations
      dL(t)  = -L(t)B(t)dt,        L(T)  = L
      dM⁺(t) = -L(t)a(t)L(t)'dt,   M⁺(T) = Σ
      dμ(t)  = -L(t)β(t)dt,        μ(T)  = 0
"""

struct GuidedProposal{TT,TA} <: ContinuousTimeProcess{ℝ{3}}
    ξ
    Target::TT
    Auxiliary::TA
    tt::Vector{Float64}
    L
    M
    μ

    function GuidedProposal(ξ, Target::TT, Auxiliary::TA, t, Σ, L) where {TT,TA<:ContinuousTimeProcess{ℝ{3}}}
        tt = collect(t)
        Lt = zeros(typeof(L), length(tt))
        Lt[end] = L
        M⁺ = Σ
        Mt = zeros(typeof(Σ), length(tt))
        Mt[end] = inv(Σ)
        μt = zeros(typeof(ξ), length(tt))
        μ = μt[end]
        for i in length(tt)-1:-1:1
            dt = tt[i] - tt[i+1]
            L = kernelr3((t, y, ℙt) -> -y*Bridge.B(t, ℙt), tt[i+1], L, dt, Auxiliary)
            M⁺ = kernelr3((t, y, (L,ℙt)) -> -outer(L*Bridge.σ(t, ℙt)), tt[i+1], M⁺, dt, (L,Auxiliary))
            μ = kernelr3((t, y, (L,ℙt)) -> -L*Bridge.β(t, ℙt), tt[i+1], μ, dt, (L,Auxiliary))

            Lt[i] = L
            Mt[i] = inv(M⁺)
            μt[i] = μ
        end
        new{TT, TA}(ξ, Target, Auxiliary, tt, Lt, Mt, μt)
    end
end

"""
    Settings for the guided proposal
        dXtᵒ = b(t, Xtᵒ)dt + a(t, Xtᵒ)r(t, Xtᵒ)dt + σ(t, Xtᵒ)dWt
"""

Htilde((i, t)::IndexedTime, x, ℙᵒ::GuidedProposal) = ℙᵒ.L[i]'*ℙᵒ.M[i]*ℙᵒ.L[i]
rtilde((i, t)::IndexedTime, x, ℙᵒ::GuidedProposal) = ℙᵒ.L[i]'*ℙᵒ.M[i]*(ℙᵒ.ξ .- ℙᵒ.μ[i] .- ℙᵒ.L[i]*x)

function Bridge.b(t, x, ℙᵒ::GuidedProposal)
    k = findmin(abs.(ℙᵒ.tt.-t))[2]
    ℙ = ℙᵒ.Target
    ℙt = ℙᵒ.Auxiliary
    a = Bridge.σ(t, x, ℙ)*Bridge.σ(t, x, ℙ)'
    return Bridge.b(t, x, ℙ) + a*rtilde((k, ℙᵒ.tt[k]), x, ℙᵒ)
end

Bridge.σ(t, x, ℙᵒ::GuidedProposal) = Bridge.σ(t, x, ℙᵒ.Target)
Bridge.constdiff(::GuidedProposal) = false

"""
    Calculating the factors in the radon-Nikodym derivative
        dℙˣ/dℙᵒ = ptilde/p ψ(Xᵒ)
"""

# Calculate ψ(Xᵒ) where Xᵒ is a guided proposal (sample from ℙᵒ)
function logψ(::LeftRule, Xᵒ::T, ℙᵒ::GuidedProposal) where {T<:SamplePath{ℝ{3}}}
    tt = Xᵒ.tt
    yy = Xᵒ.yy

    ℙ = ℙᵒ.Target
    ℙt = ℙᵒ.Auxiliary

    out::eltype(ℙᵒ.ξ) = 0.
    for i in 1:(length(tt)-1)
        dt = tt[i+1]-tt[i]
        s = tt[i]
        y = yy[i]
        r = rtilde((i,s), y, ℙᵒ)

        out += dot(Bridge.b(s, y, ℙ) - Bridge.b(s, y, ℙt) , r)*dt
        if !Bridge.constdiff(ℙᵒ)
            out -= .5*tr( (Bridge.a(s, y, ℙ) - Bridge.a(s, ℙt))*(Htilde((i,s), y, ℙᵒ) - r*r') )*dt
        end
    end
    return out
end

# Transform a wiener process W into a sample Xᵒ from ℙᵒ starting at x₀
# while simultaneously calculating logψ(Xᵒ) of the resulting sample.
function logψ!(::LeftRule, ::Ito, W, x₀, ℙᵒ::GuidedProposal)
    tt = W.tt
    ww = W.yy
    Xᵒ = deepcopy(W)
    Xᵒ.yy[1] = x₀

    ℙ = ℙᵒ.Target
    ℙt = ℙᵒ.Auxiliary

    x = copy(x₀)
    out::eltype(x₀) = 0.
    for i in 1:(length(tt)-1)
        dt = tt[i+1] - tt[i]
        dw = ww[i+1] - ww[i]
        s = tt[i]
        y = yy[i]
        r = rtilde((i,s), y, ℙᵒ)

        # terms for logψ
        out += dot(Bridge.b(s, y, ℙ) - Bridge.b(s, y, ℙt) , r)*dt
        if !Bridge.constdiff(ℙᵒ)
            out -= .5*tr( (Bridge.a(s, y, ℙ) - Bridge.a(s, ℙt))*(Htilde((i,s), y, ℙᵒ) - r*r') )*dt
        end

        # Simulating the next step of Xᵒ
        y .= y + Bridge.b(s, y, ℙᵒ)*dt + Bridge.σ(s, y, ℙᵒ)*dw
        Xᵒ.yy[i+1] = y
    end
    W = Xᵒ
    out
end

function logψ!(::LeftRule, ::Stratonovich, W, x₀, ℙᵒ::GuidedProposal)
    tt = W.tt
    ww = W.yy
    Xᵒ = deepcopy(W)
    Xᵒ.yy[1] = x₀
    y = x₀

    ℙ = ℙᵒ.Target
    ℙt = ℙᵒ.Auxiliary

    x = copy(x₀)
    out::eltype(x₀) = 0.
    for i in 1:(length(tt)-1)
        dt = tt[i+1] - tt[i]
        dw = ww[i+1] - ww[i]
        s = tt[i]
        r = rtilde((i,s), y, ℙᵒ)

        # terms for logψ
        out += dot(Bridge.b(s, y, ℙ) - Bridge.b(s, y, ℙt) , r)*dt
        if !Bridge.constdiff(ℙᵒ)
            out -= .5*tr( (Bridge.a(s, y, ℙ) - Bridge.a(s, ℙt))*(Htilde((i,s), y, ℙᵒ) - r*r') )*dt
        end

        # Simulating the next step of Xᵒ
        yᴱ = y + Bridge.b(s, y, ℙᵒ)*dt + Bridge.σ(s, y, ℙᵒ)*dw
        y = y + Bridge.b(s, y, ℙᵒ)*dt + .5*(Bridge.σ(s+dt, yᴱ,ℙᵒ) + Bridge.σ(s, y, ℙᵒ))*dw
        Xᵒ.yy[i+1] = y
    end
    W = Xᵒ
    out
end

# log(ptilde)
function lptilde((i,t)::IndexedTime, x, ℙᵒ::GuidedProposal)
    μ = ℙᵒ.μ[i]
    L = ℙᵒ.L[i]
    M = ℙᵒ.M[i]
    return logpdf(MvNormalCanon(M*(μ+L*x) , Matrix(Hermitian(M))) , ℙᵒ.ξ)
end

# log(p) = log(ptilde) + ∫ ψ(Xᵒ) dℙᵒ(Xᵒ) can be approximated using a sample Xᵒ from ℙᵒ
function logp(::LeftRule, Xᵒ::T, x₀, ℙᵒ::GuidedProposal) where {T<:SamplePath{ℝ{3}}}
    lptilde((1,0.), x₀ , ℙᵒ) + logψ(LeftRule(), Xᵒ, ℙᵒ)
end

function logp(::LeftRule, Xᵒ::T, x₀, ℙᵒ::GuidedProposal) where {T<:SamplePath{ℝ{3}}}
    lptilde((1,0.), x₀ , ℙᵒ) + logψ(LeftRule(), Xᵒ, ℙᵒ)
end

function logp!(::LeftRule, ::Ito, W::T, x₀, ℙᵒ::GuidedProposal) where {T<:SamplePath{ℝ{3}}}
    lptilde((1,0.), x₀, ℙᵒ) + logψ!(LeftRule(), Ito(), W, x₀, ℙᵒ)
end

function logp!(::LeftRule, ::Stratonovich, W::T, x₀, ℙᵒ::GuidedProposal) where {T<:SamplePath{ℝ{3}}}
    lptilde((1,0.), x₀, ℙᵒ) + logψ!(LeftRule(), Stratonovich(), W, x₀, ℙᵒ)
end
