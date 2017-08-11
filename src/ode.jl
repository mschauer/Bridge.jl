
"""
    kernelrk(f, t, y, dt, P)

Ralston (1965) update (order 3 step of the Bogacki–Shampine 1989 method)
to solve ``y(t + dt) - y(t) = \int_t^{t+dt} f(s, y(s)) ds``.
"""
function kernelrk(f, t, y, dt, P)
    k1 = f(t, y, P)
    k2 = f(t + 1/2*dt, y + 1/2*dt*k1, P)
    k3 = f(t + 3/4*dt, y + 3/4*dt*k2, P)
    y + dt*(2/9*k1 + 1/3*k2 + 4/9*k3)
end

@inline _dK(t, K, P) = B(t, P)*K + K*B(t, P)' - a(t, P)
@inline _F(t, v, P) = B(t, P)*v + β(t, P)

"""
    gpK!(K::SamplePath, P)

Precompute ``K = H^{-1}`` from ``(d/dt)K = BK + KB' + a`` for a guided proposal.
"""
gpK!(K::SamplePath{T}, P) where {T} = _solvebackward!(_dK, K, zero(T), P)
gpV!(V::SamplePath{T}, v::T, P) where {T} = _solvebackward!(_F, V, v, P)

function solvebackward!(F, X, xT, P) 
     _solvebackward!(F, X, xT, P) 
end
@inline function _solvebackward!(F, X::SamplePath{T}, xT, P) where {T}
    tt = X.tt
    yy = X.yy
    yy[end] = y::T = xT
    for i in length(tt)-1:-1:1
        y = kernelrk(F, tt[i+1], y, tt[i] - tt[i+1], P)  
        yy[i] = y  
    end
    X
end

"""
    solve!(F, X::SamplePath, x0, P) -> X

Solve ordinary differential equation `(d/dx) x(t) = F(t, x(t), P)` on the fixed
grid `X.tt` writing into `X.yy` using a non-adaptive Ralston (1965) update (order 3).
Call `_solve!` to inline.

"Pretty fast if `x` is a bitstype or a StaticArray."

Todo: use Bogacki–Shampine method to give error estimate.
"""
function solve!(F, X, x0, P) 
     _solve!(F, X, x0, P) 
end
@inline function _solve!(F, X::SamplePath{T}, x0, P) where {T}
    tt = X.tt
    yy = X.yy
    yy[1] = y::T = x0
    for i in 2:length(tt)
        y = kernelrk(F, tt[i-1], y, tt[i] - tt[i-1], P)  
        yy[i] = y  
    end
    X
end
