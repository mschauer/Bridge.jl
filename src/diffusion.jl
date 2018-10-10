
"""
    lp(s, x, t, y, P)

Log-transition density, shorthand for `logpdf(transitionprob(s,x,t,P),y)`.
"""
lp(s, x, t, y, P) = logpdf(transitionprob(s,x,t,P),y)


"""
    llikelihood(X::SamplePath, P::ContinuousTimeProcess)

Log-likelihood of observations `X` using transition density `lp`.
"""
function llikelihood(X::SamplePath, P::ContinuousTimeProcess)
    ll = 0.
    for i in 2:length(X.tt)
        ll += lp(X.tt[i-1], X.yy[i-1], X.tt[i], X.yy[i], P)
    end
    ll
end


"""
    sample(tt, P, x1=zero(T))

Sample the process `P` on the grid `tt` exactly from its `transitionprob`(-ability)
starting in `x1`.
"""
sample(tt, P::ContinuousTimeProcess{T}, x1=zero(T)) where {T} =
    sample!(samplepath(tt, zero(T), P, x1))

"""
    sample!([::Bridge.TransitionProb], X, P, x1=zero(T))

Sample the process `P` on the grid `X.tt` exactly from its `transitionprob`(-ability)
starting in `x1` writing into `X.yy`.
"""
sample!(X, P::ContinuousTimeProcess{T}, x1=zero(T)) where {T} = sample!(TransitionProb(), X, P, x1)

struct TransitionProb
end

function sample!(::TransitionProb, X, P::ContinuousTimeProcess{T}, x1) where T
    tt = X.tt
    yy = X.yy
    x = convert(T, x1)
    yy[1] = x
    for i in 2:length(tt)
        x = rand(transitionprob(tt[i-1], x, tt[i], P))
        yy[i] = x
    end
    X
end


"""
    quvar(X)

Computes the (realized) quadratic variation of the path `X`.
"""
function quvar(X::SamplePath{T}) where T
        s = outer(zero(T))
        for u in diff(X.yy)
            s += outer(u)
        end
        s
end


"""
    bracket(X)
    bracket(X,Y)

Computes quadratic variation process of `X` (of `X` and `Y`).
"""
function bracket(X::SamplePath)
        cumsum0(outer.(diff(X.yy)))
end

function bracket(X::SamplePath,Y::SamplePath)
        cumsum0(outer.(diff(X.yy),diff(X.yy)))
end


"""
    ito(Y, X)

Integrate a stochastic process `Y` with respect to a stochastic differential `dX`.
"""
function ito(X::SamplePath, W::SamplePath{T}) where T
        @assert(X.tt[1] == W.tt[1])
        n = length(X)
        yy = similar(W.yy, n)
        yy[1] = zero(T)
        for i in 2:n
                @assert(X.tt[i] == W.tt[i])
                yy[i] = yy[i-1] + X.yy[i-1]*(W.yy[i]-W.yy[i-1])
        end
        SamplePath{T}(X.tt, yy)
end


"""
    girsanov(X::SamplePath, P::ContinuousTimeProcess, Pt::ContinuousTimeProcess)

Girsanov log likelihood ``\\mathrm{d}P/\\mathrm{d}Pt(X)``
"""
function girsanov(X::SamplePath{T}, P::ContinuousTimeProcess{T}, Pt::ContinuousTimeProcess{T}) where T
    tt = X.tt
    xx = X.yy

    som::Float64 = 0.
    for i in 1:length(tt)-1 #skip last value, summing over n-1 elements
      s = tt[i]
      x = xx[i]
      B = Bridge.b(s,x, P)
      Bt = Bridge.b(s,x, Pt)
      DeltaBG = Bridge.Γ(s, x, P)*(B-Bt)
      som += dot(DeltaBG, xx[i+1]-xx[i] - 0.5(B + Bt) * (tt[i+1]-tt[i]))
    end
    som
end


"""
    NoDrift(tt, P)

As `P`, but without drift.
"""
struct NoDrift{S,T} <: ContinuousTimeProcess{T}
    P::S
    NoDrift(P::S) where {S<:ContinuousTimeProcess{T}} where{T} = new{S,T}(P)
end

b(t, x, P::NoDrift) = zero(valtype(P))
bderiv(t, x, P::NoDrift) = zero(outertype(P))
σ(t, x, P::NoDrift) = σ(t, x, P.P)
