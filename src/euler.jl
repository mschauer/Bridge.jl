euler(u, W, P) = euler!(copy(W), u, W, P)
function euler!{T}(Y, u, W::SamplePath{T}, P)

    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    ww = W.yy
    tt = Y.tt
    yy = Y.yy
    tt[:] = W.tt

    y = u

    for i in 1:N-1
        yy[.., i] = y
        y = y + b(tt[i], y, P)*(tt[i+1]-tt[i]) + σ(tt[i], y, P)*(ww[.., i+1]-ww[..,i])
    end
    yy[.., N] = y
    SamplePath{T}(tt, yy)
end

# euler for bridges starting from P.v0 to P.v1 

eulerb(W, P) = eulerb!(copy(W), W, P)
function eulerb!{T}(Y, W::SamplePath{T}, P)

    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    ww = W.yy
    tt = Y.tt
    
    tt[1] != P.t0 && error("time axis mismatch between W and P  ")
    tt[end] != P.t1 && error("time axis mismatch between W and P  ")
    
    yy = Y.yy
    tt[:] = W.tt

    y = P.v0

    for i in 1:N-1
        yy[.., i] = y
        y = y + b(tt[i], y, P)*(tt[i+1]-tt[i]) + σ(tt[i], y, P)*(ww[.., i+1]-ww[..,i])
    end
    yy[.., N] = P.v1
    SamplePath{T}(tt, yy)
end


innovations(Y, P) = innovations!(copy(Y), Y, P)
function innovations!{T}(W, Y::SamplePath{T}, P)

    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    yy = Y.yy
    tt = Y.tt
    ww = W.yy
    W.tt[:] = Y.tt

    w = zero(ww[.., 1])

    for i in 1:N-1
        ww[.., i] = w
        w = w + σ(tt[i], yy[.., i], P)\(yy[.., i+1] - yy[.., i] - b(tt[i], yy[.., i], P)*(tt[i+1]-tt[i])) 
    end
    ww[.., N] = w
    SamplePath{T}(tt, ww)
end
