
"""
Inplace sde solver.

Typical call is like
solve!(EulerMaruyama!(), X, x0, W, P)
or
solve!(StratonovichHeun!(), X, x0, W, P)

Here
- x0 is the starting point,
- W the driving Wiener process,
- P the ContinuousTimeProcess

Requires
- drift function _b!((i,t¯), y, tmp1, P)
- diffusion function σ!(t¯, y, dw, tmp2, P)

The result is written into X
"""
function solve!(solver::EulerMaruyama!, Y::VSamplePath, u::T, W, P::ProcessOrCoefficients) where {T}
    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    tt = Y.tt
    tt[:] = W.tt
    yy = Y.yy
    y::T = copy(u)

    size(Y.yy) != (length(y), N) && error("Starting point has wrong length.")
    #assert(size(W.yy) == (length(y), N))
    tmp1 = copy(y)
    tmp2 = copy(y)
    dw = W.yy[.., 1]
    dw2 = copy(dw)
    for i in 1:N-1
        t¯ = tt[i]
        dt = tt[i+1] - t¯
        for k in eachindex(tmp1)
            @inbounds yy[k, i] = y[k]
        end
        for k in eachindex(dw)
            @inbounds dw[k] = W.yy[k, i+1] - W.yy[k, i]
        end
        _b!((i,t¯), y, tmp1, P)
        σ!(t¯, y, dw, tmp2, P)
        for k in eachindex(y)
            @inbounds y[k] = y[k] + tmp1[k]*dt + tmp2[k]
        end
    end
    yy[.., N] = endpoint(y, P)
    Y
end

function Bridge.solve!(solver::EulerMaruyama!, Y::SamplePath, u, W::SamplePath, P::Bridge.ProcessOrCoefficients)
    N = length(W)

    tt = W.tt
    yy = Y.yy
    copyto!(yy[1], u)

    i = 1
    dw = W.yy[i+1] - W.yy[i]
    t¯ = tt[i]
    dt = tt[i+1] - t¯

    tmp1 = Bridge._b!((i,t¯), u, 0*u, P)
    tmp2 = Bridge.σ(t¯, u, dw, P)

    y = u + tmp1*dt + tmp2



    for i in 2:N-1
        t¯ = tt[i]
        dt = tt[i+1] - t¯
        copyto!(yy[i], y)
        if dw isa Number
            dw = W.yy[i+1] - W.yy[i]
        else
            for k in eachindex(dw)
                dw[k] = W.yy[i+1][k] - W.yy[i][k]
            end
        end

        Bridge._b!((i,t¯), y, tmp1, P)
        Bridge.σ!(t¯, y, dw, tmp2, P)

        for k in eachindex(y)
            y[k] = y[k] + tmp1[k]*dt + tmp2[k]
        end
    end
    copyto!(yy[end], Bridge.endpoint(y, P))
    SamplePath(tt, yy)
end
