"""
    R3!

Inplace Ralston (1965) update (order 3 step of the Bogacki–Shampine 1989 method)
to solve ``y(t + dt) - y(t) = \\int_t^{t+dt} F(s, y(s)) ds``.
"""
struct R3! <: ODESolver
end

workspace(::Bridge.R3!, y) = (copy(y), copy(y), copy(y), copy(y))

"""
One step for inplace Ralston (1965) update (order 3 step of the Bogacki–Shampine 1989 method)
to solve ``y(t + dt) - y(t) = \\int_t^{t+dt} f(s, y(s)) ds``.
Starting point is specified by (t,y)

f!(t,y,k) is a function that takes (t,y) and writes the result in k.
ws contains 4 copies of the type of y
the result is written into out which is of type y
"""
function kernelr3!(f!, t, y, ws, out, dt)
    y2, k1, k2, k3 = ws
    f!(t, y, k1)
    y2 .= y + 1/2*dt*k1
    f!(t + 1/2*dt, y2, k2)
    y2 .= y + 3/4*dt*k2
    f!(t + 3/4*dt, y2, k3)
    out .= y + dt*(2/9*k1 + 1/3*k2 + 4/9*k3)
end

function solvebackward!(::Bridge.R3!, f!, X, xT)
    tt = X.tt
    n = length(tt)
    yy = X.yy
    y = copy(xT)
    yy[.., n] = y
    ws = (copy(y), copy(y), copy(y), copy(y)) # y2, k1, k2, k3
    for i in n-1:-1:1
        kernelr3!(f!, tt[i+1], y, ws, y, tt[i] - tt[i+1])
        yy[.., i] = y
    end
    X
end

function solvebackward!(::Bridge.R3!, f!, X::SamplePath, xT)
    tt = X.tt
    n = length(tt)
    yy = X.yy
    y = copy(xT)
    z = yy[n]
    z .= y
    ws = (copy(y), copy(y), copy(y), copy(y)) # y2, k1, k2, k3
    for i in n-1:-1:1
        kernelr3!(f!, tt[i+1], y, ws, y, tt[i] - tt[i+1])
        z = yy[i]
        z .= y
    end
    X
end
