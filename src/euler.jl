
# Allow `UniformScaling`s as value for σ
_scale(w, σ) = σ*w
_scale(w::Number, σ::UniformScaling) = σ.λ*w

"""
    SDESolver

Abstract (super-)type for solving methods for stochastic differential equations.
"""
abstract type SDESolver
end

struct EulerMaruyama <: SDESolver
end
const Euler = EulerMaruyama

"""
    EulerMaruyama() <: SDESolver

Euler-Maruyama scheme. `Euler` is defined as alias.
"""
Euler, EulerMaruyama





"""
    StochasticHeun() <: SDESolver

Stochastic heun scheme.
"""
struct StochasticHeun <: SDESolver
end

"""
    Mdb() <: SDESolver

Euler scheme with the diffusion coefficient correction of the modified diffusion bridge.
"""
struct Mdb <: SDESolver
end


"""
    StochasticRungeKutta() <: SDESolver

Stochastic Runge-Kutta scheme for `T<:Number`-valued processes.
"""
struct StochasticRungeKutta <: SDESolver
end

struct EulerMaruyama! <: SDESolver
end

endpoint(y, P) = y


function solve!(::StochasticHeun, Y, u, W::SamplePath, P::ProcessOrCoefficients)

    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    ww = W.yy
    tt = Y.tt
    yy = Y.yy
    tt[:] = W.tt

    y = u

    for i in 1:N-2 # fix me
        yy[.., i] = y
        B = _b((i,tt[i]), y, P)
        y2 = y + B*(tt[i+1]-tt[i])
        y = y + 0.5*(_b((i+1,tt[i+1]), y2, P) + B)*(tt[i+1]-tt[i]) + σ(tt[i], y, P)*(ww[.., i+1]-ww[..,i])
    end
    yy[.., N-1] = endpoint(y, P)
    Y
end





"""
    solve(method::SDESolver, u, W::SamplePath, P) -> X
    solve(method::SDESolver, u, W::SamplePath, (b, σ)) -> X

Solve stochastic differential equation ``dX_t = b(t,X_t)dt + σ(t,X_t)dW_t``
using `method` in place.

# Example

```
solve(EulerMaruyama(), 1.0, sample(0:0.1:10, Wiener()), ((t,x)->-x, (t,x)->I))
```

```
struct OU <: ContinuousTimeProcess{Float64}
    μ::Float64
end
Bridge.b(s, x, P::OU) = -P.μ*x
Bridge.σ(s, x, P::OU) = I

solve(EulerMaruyama(), 1.0, sample(0:0.1:10, Wiener()), OU(1.4))
```
"""
solve(method::SDESolver, u::T, W::SamplePath, P::ProcessOrCoefficients) where {T} =
    solve!(method, samplepath(W.tt, zero(u)), u, W, P)

"""
    solve(method::SDESolver, u, W::VSamplePath, P) -> X

Solve stochastic differential equation ``dX_t = b(t,X_t)dt + σ(t,X_t)dW_t``
using `method`.
"""
solve(method::SDESolver, u, W::VSamplePath{T}, P::ProcessOrCoefficients) where {T} =
    solve!(method, VSamplePath(W.tt, zeros(T, size(u)..., length(W.tt))), u, W, P)

"""
    solve!(::EulerMaruyama, Y, u, W, P) -> X

Solve stochastic differential equation ``dX_t = b(t,X_t)dt + σ(t,X_t)dW_t``
using the Euler-Maruyama scheme in place.
"""
function solve!(::EulerMaruyama, Y, u::T, W::SamplePath, P::ProcessOrCoefficients) where {T}
    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    ww = W.yy
    tt = Y.tt
    yy = Y.yy
    tt[:] = W.tt

    y::T = u

    for i in 1:N-1
        yy[.., i] = y
        y = y + _b((i,tt[i]), y, P)*(tt[i+1]-tt[i]) + _scale((ww[.., i+1]-ww[..,i]), σ(tt[i], y, P))
    end
    yy[.., N] = endpoint(y, P)
    Y
end

# fallback method
function solve!(::EulerMaruyama, Y, u::T, W::AbstractPath, P::ProcessOrCoefficients) where {T}
    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    ww = W.yy
    tt = Y.tt
    yy = Y.yy
    tt[:] = W.tt

    y::T = u

    for (i, t, dt, dw) in increments(W)
        yy[.., i] = y
        y = y + _b((i,t), y, P)*dt + _scale(dw, σ(t, y, P))
    end
    yy[.., N] = endpoint(y, P)
    Y
end



"""
    solve(method, W, P) -> Y

Integrate with `method`, where `P` is a bridge proposal from
`startpoint(P)`` to `endpoint(P)`.

# Examples

```
cs = Bridge.CSpline(tt[1], tt[end], Bridge.b(tt[1], v[1], P),  Bridge.b(tt[end], v[2], P))
P° = BridgeProp(Pσ, v), Pσ.a, cs)
W = sample(tt, Wiener())
solve(Euler(), W, P°)
```
"""
solve(method, W::SamplePath, P) = let u = startpoint(P), Y = samplepath(W.tt, zero(u)); solve!(method, Y, u, W, P); Y end
solve!(method, Y::SamplePath, W::SamplePath, P::ContinuousTimeProcess) = solve!(method, Y, startpoint(P), W, P)


#### Guided Bridges
endpoint(y, P::GuidedBridge) =
 norm(P.H♢[end], 1) < eps() ? P.V[end] : y



solve(::Euler, u, W::SamplePath, P::Union{GuidedBridge,PartialBridge,PartialBridgeνH}) = let X = samplepath(W.tt, zero(u)); solve!(Euler(), X, u, W, P); X end
function solve!(::Euler, Y, u, W::SamplePath, P::Union{GuidedBridge,PartialBridge,PartialBridgeνH})
    W.tt === P.tt && error("Time axis mismatch between bridge P and driving W.") # not strictly an error

    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    ww = W.yy
    tt = Y.tt
    yy = Y.yy
    tt[:] = P.tt

    y::typeof(u) = u
    if typeof(u) != valtype(P)
    #    @warn "Starting point not of valtype." maxlog=10
    end
    for i in 1:N-1
        yy[.., i] = y
        y = y + _b((i, tt[i]), y, P)*(tt[i+1]-tt[i]) + _scale((ww[.., i+1]-ww[..,i]), σ(tt[i], y, P))
    end
    yy[.., N] = endpoint(y, P)
    yy[.., N]
end


####


"""
    solve!(method, Y, W, P) -> Y

Integrate with `method`, where `P is a bridge proposal overwriting `Y`.
"""
solve!


function solve!(::Mdb, Y, u, W::SamplePath, P::ContinuousTimeProcess{T}) where {T}
    W.tt === P.tt && error("Time axis mismatch between bridge P and driving W.") # not strictly an error

    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    ww = W.yy
    tt = Y.tt
    yy = Y.yy
    tt[:] = P.tt

    y::T = u

    for i in 1:N-1
        yy[.., i] = y
        y = y + _b((i, tt[i]), y, P)*(tt[i+1]-tt[i]) + _scale((ww[.., i+1]-ww[..,i]), σ(tt[i], y, P)*sqrt((tt[end]-tt[i+1])/(tt[end]-tt[i])))
    end
    yy[.., N] = endpoint(y, P)
    Y
end

function solve!(::StochasticRungeKutta, Y, u::T, W::SamplePath, P::ProcessOrCoefficients) where T<:Number

    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    ww = W.yy
    tt = Y.tt
    yy = Y.yy
    tt[:] = W.tt

    y = u

    for i in 1:N-1
        yy[.., i] = y
        delta = tt[i+1]-tt[i]
        sqdelta = sqrt(delta)
        B = _b((i,tt[i]), y, P)
        S = σ(tt[i], y, P)
        dw = ww[.., i+1]-ww[..,i]
        y = y + B*delta + S*dw
        ups = y + B*delta + S*sqdelta
        y = y + 0.5(σ(tt[i+1], ups, P) - S)*(dw^2 - delta)/sqdelta

    end
    yy[.., N] = y
    SamplePath{T}(tt, yy)
end

innovations(method, Y, P) = innovations!(method, copy(Y), Y, P)
function innovations!(::EulerMaruyama, W, Y::SamplePath, P)

    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    yy = Y.yy
    tt = Y.tt
    ww = W.yy
    W.tt[:] = Y.tt

    w = zero(ww[.., 1])

    for i in 1:N-1
        ww[.., i] = w
        w = w + inv(σ(tt[i], yy[.., i], P))*(yy[.., i+1] - yy[.., i] - _b((i,tt[i]), yy[.., i], P)*(tt[i+1]-tt[i]))
    end
    ww[.., N] = w
    W
end

function innovations!(::Mdb, W, Y::SamplePath, P)

    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    yy = Y.yy
    tt = Y.tt
    ww = W.yy
    W.tt[:] = Y.tt

    w = zero(ww[.., 1])

    for i in 1:N-1
        ww[.., i] = w
        w = w + sqrt((tt[end]-tt[i+1])/(tt[end]-tt[i]))\inv(σ(tt[i], yy[.., i], P))*(yy[.., i+1] - yy[.., i] - _b((i,tt[i]), yy[.., i], P)*(tt[i+1]-tt[i]))
    end
    ww[.., N] = w
    W
end
