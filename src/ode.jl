"""
    ODESolver

Abstract (super-)type for solving methods for ordinary differential equations.
"""
abstract type ODESolver 
end

"""
    R3

Ralston (1965) update (order 3 step of the Bogacki–Shampine 1989 method)
to solve ``y(t + dt) - y(t) = \\int_t^{t+dt} f(s, y(s)) ds``.
"""
struct R3 <: ODESolver
end

"""
    BS3

Ralston (1965) update (order 3 step of the Bogacki–Shampine 1989 method)
to solve ``y(t + dt) - y(t) = \\int_t^{t+dt} f(s, y(s)) ds``. Uses Bogacki–Shampine method 
to give error estimate. 
"""
struct BS3 <: ODESolver
end

function kernelr3(f, t, y, dt, P)
    k1 = f(t, y, P)
    k2 = f(t + 1/2*dt, y + 1/2*dt*k1, P)
    k3 = f(t + 3/4*dt, y + 3/4*dt*k2, P)
    y + dt*(2/9*k1 + 1/3*k2 + 4/9*k3)
end

function kernelbs3(f, t, y, dt, P, k = f(t, y, P))
    k1 = k
    k2 = f(t + 1/2*dt, y + 1/2*dt*k1, P)
    k3 = f(t + 3/4*dt, y + 3/4*dt*k2, P)
    yº = y + dt*(2/9*k1 + 1/3*k2 + 4/9*k3)
    k4 = f(t + dt, yº, P)
    err = dt*(-5/72*k1 + 6/72*k2 + 8/72*k3 - 9/72*k4)
    yº, k4, err
end

@inline _dK(t, K, P) = B(t, P)*K + K*B(t, P)' - a(t, P)
@inline _F(t, v, P) = B(t, P)*v + β(t, P)

"""
    gpK!(K::SamplePath, P)

Precompute ``K = H^{-1}`` from ``(d/dt)K = BK + KB' + a`` for a guided proposal.
"""
gpK!(K::SamplePath{T}, P) where {T} = _solvebackward!(R3(), _dK, K, zero(T), P)
gpV!(V::SamplePath{T}, v::T, P) where {T} = _solvebackward!(R3(), _F, V, v, P)

function solvebackward!(method, F, X, xT, P) 
     _solvebackward!(method, F, X, xT, P) 
end
@inline function _solvebackward!(::R3, F, X::SamplePath{T}, xT, P) where {T}
    tt = X.tt
    yy = X.yy
    yy[end] = y::T = xT
    for i in length(tt)-1:-1:1
        y = kernelr3(F, tt[i+1], y, tt[i] - tt[i+1], P)  
        yy[i] = y  
    end
    X
end

"""
    solve!(method, X::SamplePath, x0, P) -> X, [err]

Solve ordinary differential equation ``(d/dx) x(t) = F(t, x(t), P)`` on the fixed
grid `X.tt` writing into `X.yy` 

`method::R3` - using a non-adaptive Ralston (1965) update (order 3).

`method::BS3` use non-adaptive Bogacki–Shampine method to give error estimate.

Call `_solve!` to inline. "Pretty fast if `x` is a bitstype or a StaticArray."

"""
function solve!(method, F, X, x0, P) 
     _solve!(method, F, X, x0, P) 
end
@inline function _solve!(::R3, F, X::SamplePath{T}, x0, P) where {T}
    tt = X.tt
    yy = X.yy
    yy[1] = y::T = x0
    for i in 2:length(tt)
        y = kernelr3(F, tt[i-1], y, tt[i] - tt[i-1], P)  
        yy[i] = y  
    end
    X
end
@inline function _solve!(::BS3, F, X::SamplePath{T}, x0, P) where {T}
    tt = X.tt
    yy = X.yy
    0 < length(tt) || throw(ArgumentError("length(X) == 0"))
    yy[1] = y::T = x0   
    length(tt) == 1 && return X, 0.0
    y, k, e = kernelbs3(F, tt[1], y, tt[2] - tt[1], P)
    yy[2] = y
    err = norm(e, 1)
    for i in 3:length(tt)
        y, k, e = kernelbs3(F, tt[i-1], y, tt[i] - tt[i-1], P, k)  
        err = err + norm(e, 1)
        yy[i] = y  
    end
    X, err
end

