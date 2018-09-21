#=
function solve!(::EulerMaruyama, Y, u, W::SamplePath, P::ProcessOrCoefficients)
    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    tt, yy = Y.tt, Y.yy
    tt .= W.tt
    ww = W.yy
 
    y::typeof(u) = u

    for i in 1:N-1
        dt = tt[i+1] - tt[i]
        yy[i] = y
        y = y + b(tt[i], y, P)*dt + _scale((ww[i+1]-ww[i]), σ(tt[i], y,  P))
    end
    yy[N] = y
    Y
end
=#

"""
Currently only timedependent sigma, as Ito correction is necessary
"""
function solvebackward!(::EulerMaruyama, Y, u, W::SamplePath, P::ProcessOrCoefficients)
    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    tt, yy = Y.tt, Y.yy
    tt .= W.tt
    ww = W.yy

    y::typeof(u) = u

    for i in N:-1:2
        dt = tt[i-1] - tt[i]
        yy[i] = y
        y = y + b(tt[i], y, P)*dt + _scale((ww[i-1]-ww[i]), σ(tt[i], P))
    end
    yy[1] = y
    Y
end