euler(u, W, P) = euler!(copy(W), u, W, P)
function euler!(Y, u, W::SamplePath, P)

    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    ww = W.yy
    tt = Y.tt
    yy = Y.yy
    tt[:] = W.tt

    y = u

    for i in 1:N-1
        yy[.., i] = y
        factor = σ(tt[i], y, P)
        isa(factor,UniformScaling) && ( factor = factor.λ )
        y = y + b(tt[i], y, P)*(tt[i+1]-tt[i]) + factor*(ww[.., i+1]-ww[..,i])
    end
    yy[.., N] = y
    Y
end

heun(u, W, P) = heun!(copy(W), u, W, P)
function heun!(Y, u, W::SamplePath, P)

    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    ww = W.yy
    tt = Y.tt
    yy = Y.yy
    tt[:] = W.tt

    y = u

    for i in 1:N-2 #f ix me
        yy[.., i] = y
        B = b(tt[i], y, P)
        y2 = y + B*(tt[i+1]-tt[i]) 
        y = y + 0.5*(b(tt[i+1], y2, P) + B)*(tt[i+1]-tt[i]) + σ(tt[i], y, P)*(ww[.., i+1]-ww[..,i])
    end
    yy[.., N-1] = y
    Y
end

"""
    thetamethod
  
Solve stochastic differential equation using the theta method and Newton-Raphson steps
"""
thetamethod(u, W, P, theta=0.5) = thetamethod!(copy(W), u, W, P, theta)
function thetamethod!(Y, u, W::SamplePath, P, theta=0.5)

    N = length(W)
    N != length(Y) && error("Y and W differ in length.")
    assert(constdiff(P))

    ww = W.yy
    tt = Y.tt
    yy = Y.yy
    tt[:] = W.tt

    y = u

    for i in 1:N-2 # fix me
        yy[.., i] = y
        y2 = y
        dw = (ww[.., i+1]-ww[..,i])
        delta1 = b(tt[i], y, P)*(tt[i+1]-tt[i]) 
        local delta2
       
        ## solve y2 - y - theta*(b(tt[i+1], y2, P)*(tt[i+1]-tt[i]) - (1- theta)*(b(tt[i], y, P)*(tt[i+1]-tt[i] - σ(tt[i], y2, P)*dw = 0 with newton raphson step
        
        const eps2 = 5e-6
        const MM = 8
        for mm in 1:MM
            
            delta2 = b(tt[i+1], y2, P)*(tt[i+1]-tt[i]) 
            dy2 = -inv(I - theta*(bderiv(tt[i+1], y2, P)*(tt[i+1]-tt[i])))*(y2 - y - (1-theta)*delta1 - theta*delta2 - σ(tt[i], y, P)*dw)
            
            y2 += dy2
            if  maximum(abs.(dy2)) < eps2
                break;
            end
            
            if mm == MM
                warn("thetamethod: no convergence $i $y $y2  $dy2")
            end
        end
        y = y + (1-theta)*delta1 + theta*delta2 + σ(tt[i], y, P)*dw
    end
    yy[.., N-1] = y
    Y
end


mdb(u, W, P) = mdb!(copy(W), u, W, P)
function mdb!(Y, u, W::SamplePath, P)

    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    ww = W.yy
    tt = Y.tt
    yy = Y.yy
    tt[:] = W.tt

    y = u

    for i in 1:N-1
        yy[.., i] = y
        y = y + b(tt[i], y, P)*(tt[i+1]-tt[i]) + σ(tt[i], y, P)*sqrt((tt[end]-tt[i+1])/(tt[end]-tt[i]))*(ww[.., i+1]-ww[..,i])
    end
    yy[.., N] = y
    Y
end

bridge(W, P, scheme! = euler!) = bridge!(copy(W), W, P, scheme!)
function bridge!(Y, W::SamplePath, P, scheme! = euler!)
    !(W.tt[1] == P.t0 && W.tt[end] == P.t1) && error("Time axis mismatch between bridge P and driving W.") # not strictly an error
    scheme!(Y, P.v0, W, P)
    Y.yy[.., length(W.tt)] = P.v1
    Y
end

rungekutta(W, P) = rungekutta!(copy(W), W, P)
function rungekutta!{T<:Number}(Y, u, W::SamplePath{T}, P)

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

innovations(Y, P) = innovations!(copy(Y), Y, P)
function innovations!(W, Y::SamplePath, P)

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

mdbinnovations(Y, P) = mdbinnovations!(copy(Y), Y, P)
function mdbinnovations!(W, Y::SamplePath, P)

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


thetainnovations(Y::SamplePath, P, theta=0.5) = thetainnovations!(copy(Y), Y, P, theta)
function thetainnovations!(W, Y::SamplePath, P, theta=0.5)
    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    yy = Y.yy
    tt = Y.tt
    ww = W.yy
    W.tt[:] = Y.tt

    w = zero(ww[.., 1])

    for i in 1:N-2
        ww[.., i] = w
        w = w + inv(σ(tt[i], yy[.., i], P))*(yy[.., i+1] - yy[.., i] - (theta*b(tt[i+1], yy[.., i+1], P) - (1-theta)*b(tt[i], yy[.., i], P))*(tt[i+1]-tt[i])) 
    end
    ww[.., N] = ww[.., N-1] = w
    W
end


     
