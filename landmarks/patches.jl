
function Bridge.sample!(W::SamplePath{Vector{T}}, P::Wiener{Vector{T}}, y1 = W.yy[1]) where {T}
    y = copy(y1)
    copyto!(W.yy[1], y)

    for i = 2:length(W.tt)
        rootdt = sqrt(W.tt[i]-W.tt[i-1])
        for k in eachindex(y)
            y[k] =  y[k] + rootdt*randn(T)
        end
        copyto!(W.yy[i], y)
    end
    #println(W.yy[1])
    W
end


struct StratonovichHeun! <: Bridge.SDESolver
end

function Bridge.solve!(solver::StratonovichHeun!, Y::SamplePath, u, W::SamplePath, P::Bridge.ProcessOrCoefficients)
    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    tt = Y.tt
    tt[:] = W.tt
    yy = Y.yy
    y = copy(u)
    ȳ = copy(u)

    tmp1 = copy(y)
    tmp2 = copy(y)
    tmp3 = copy(y)
    tmp4 = copy(y)

    dw = copy(W.yy[1])
    for i in 1:N-1
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

        _b!((i,t¯), y, tmp1, P)
        σ!(t¯, y, dw, tmp2, P)

        for k in eachindex(y)
            ȳ[k] = y[k] + tmp1[k]*dt + tmp2[k] # Euler prediction
        end

        _b!((i + 1,t¯ + dt), ȳ, tmp3, P) # coefficients at ȳ
        σ!(t¯ + dt, ȳ, dw2, tmp4, P)

        for k in eachindex(y)
            y[k] = y[k] + 0.5*((tmp1[k] + tmp3[k])*dt + tmp2[k] + tmp4[k])
        end
    end
    copyto!(yy[end], endpoint(y, P))
    Y
end
