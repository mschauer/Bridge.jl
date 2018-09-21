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
    BridgePre() <: SDESolver

Precomputed Euler-Maruyama scheme for bridges using `bi`.
"""
struct BridgePre <: SDESolver
end




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
        B = b(tt[i], y, P)
        y2 = y + B*(tt[i+1]-tt[i])
        y = y + 0.5*(b(tt[i+1], y2, P) + B)*(tt[i+1]-tt[i]) + σ(tt[i], y, P)*(ww[.., i+1]-ww[..,i])
    end
    yy[.., N-1] = y
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
    solve!(method, SamplePath{T}(W.tt, T[zero(u) for t in W.tt]), u, W, P)

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
        y = y + b(tt[i], y, P)*(tt[i+1]-tt[i]) + _scale((ww[.., i+1]-ww[..,i]), σ(tt[i], y, P))
    end
    yy[.., N] = y
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
        y = y + b(t, y, P)*dt + _scale(dw, σ(t, y, P))
    end
    yy[.., N] = y
    Y
end



"""
    bridge(method, W, P) -> Y

Integrate with `method`, where `P` is a bridge proposal.

# Examples

```
cs = Bridge.CSpline(tt[1], tt[end], Bridge.b(tt[1], v[1], P),  Bridge.b(tt[end], v[2], P))
P° = BridgeProp(Pσ, v), Pσ.a, cs)
W = sample(tt, Wiener())
bridge(BridgePre(), W, P°)
```
"""
bridge(method::SDESolver, W, P) = bridge!(method, copy(W), W, P)
bridge!(::Euler, Y, W::SamplePath, P::ContinuousTimeProcess) = bridge!(BridgePre(), Y, W, P)

function bridge!(::BridgePre, Y, W::SamplePath, P::ContinuousTimeProcess{T}) where {T}
    W.tt === P.tt && error("Time axis mismatch between bridge P and driving W.") # not strictly an error

    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    ww = W.yy
    tt = Y.tt
    yy = Y.yy
    tt[:] = P.tt

    y::T = P.v[1]

    for i in 1:N-1
        yy[.., i] = y
        y = y + bi(i, y, P)*(tt[i+1]-tt[i]) + _scale((ww[.., i+1]-ww[..,i]), σ(tt[i], y, P))
    end
    yy[.., N] = P.v[end]
    Y
end


#### Guided Bridges

bridge(u::T, W::SamplePath, P::GuidedBridge{T}) where {T} = bridge!(samplepath(W.tt, u), W, P)
"""
    bridge!(Y, u, W, P::GuidedBridge) -> v

Integrate guided bridge proposal `P` from `u`, returning endpoint `v`.
"""
function bridge!(Y, u, W::SamplePath, P::GuidedBridge{T}) where {T}
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
        y = y + bi(i, y, P)*(tt[i+1]-tt[i]) + _scale((ww[.., i+1]-ww[..,i]), σ(tt[i], y, P))
    end
    if norm(P.H♢[end], 1) < eps()
        yy[.., N] = P.V[end]
    else
        yy[.., N] = y
    end
    yy[.., N]
end
function bridge!(Y, u, W::SamplePath, P::Union{PartialBridge{T},PartialBridgeνH{T}}) where {T}
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
        y = y + bi(i, y, P)*(tt[i+1]-tt[i]) + _scale((ww[.., i+1]-ww[..,i]), σ(tt[i], y, P))
    end
    yy[.., N] = y
    yy[.., N]
end


####


"""
    bridge!(method, Y, W, P) -> Y

Integrate with `method`, where `P is a bridge proposal overwriting `Y`.
"""
bridge!


function bridge!(::Mdb, Y, W::SamplePath, P::ContinuousTimeProcess{T}) where {T}
    W.tt === P.tt && error("Time axis mismatch between bridge P and driving W.") # not strictly an error

    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    ww = W.yy
    tt = Y.tt
    yy = Y.yy
    tt[:] = P.tt

    y::T = P.v[1]

    for i in 1:N-1
        yy[.., i] = y
        y = y + bi(i, y, P)*(tt[i+1]-tt[i]) + _scale((ww[.., i+1]-ww[..,i]), σ(tt[i], y, P)*sqrt((tt[end]-tt[i+1])/(tt[end]-tt[i])))
    end
    yy[.., N] = P.v[end]
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
        B = b(tt[i], y, P)
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
        w = w + inv(σ(tt[i], yy[.., i], P))*(yy[.., i+1] - yy[.., i] - b(tt[i], yy[.., i], P)*(tt[i+1]-tt[i]))
    end
    ww[.., N] = w
    W
end

function innovations!(::BridgePre, W, Y::SamplePath, P)

    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    yy = Y.yy
    tt = Y.tt
    ww = W.yy
    W.tt[:] = Y.tt

    w = zero(ww[.., 1])

    for i in 1:N-1
        ww[.., i] = w
        w = w + σ(tt[i], yy[.., i], P)\(yy[.., i+1] - yy[.., i] - bi(i, yy[.., i], P)*(tt[i+1]-tt[i]))
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
        w = w + sqrt((tt[end]-tt[i+1])/(tt[end]-tt[i]))\inv(σ(tt[i], yy[.., i], P))*(yy[.., i+1] - yy[.., i] - b(tt[i], yy[.., i], P)*(tt[i+1]-tt[i]))
    end
    ww[.., N] = w
    W
end
