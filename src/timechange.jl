"""
    tofs(s, T1, T2)
    soft(t, T1, T2)
  
    Time change mapping t in [T1, T2] (X-time) to s in [T1, T2]  (U-time).
"""      
tofs(s, T1, T2) = T1 + (s - T1).*(2. - (s-T1)/(T2-T1))


"""
    soft(t, T1, T2)
  
    Time change mapping s in [T1, T2] (U-time) to t in [T1, T2] (X-time), and inverse.
"""  
soft(t, T1, T2, T=T2-T1) = T2 - sqrt.(T*(T2 - t))


xofu(s, u, T1, T2, v, P) = Vs(s, T1, T2, v, P) .- (T2-s)*u
uofx(s, x, T1, T2, v, P) = (Vs(s, T1, T2, v, P) .- x)/(T2-s)
txofsu(s, u, T1, T2, v, P) = (tofs(s, T1, T2), xofu(s, u, T1, T2, v, P))

"""
    Vs (s, T1, T2, v, B, beta)

Time changed V for generation of U
"""
function Vs(s, T1, T2, v, P::LinPro, phim = expm(-P.B*(T2-T1)*(1. - (s-T1)/(T2-T1))^2))
    phim*( v .- P.μ) .-  P.μ
end
Vs(s, T1, T2, v, P::Ptilde) = V(tofs(s, T1, T2), T2, v, P)


"""
    dotVs (s, T, v, B, beta)

Time changed time derivative of V for generation of U
"""
function dotVs(s, T1, T2, v, P::LinPro, phim = expm(-P.B*(T2-T1)*(1. - (s-T1)/(T2-T1))^2))
    phim*( P.B*v .+ P.beta) 
end
dotVs(s, T1, T2, v, P::Ptilde) = dotV(tofs(s, T1, T2), T2, v, P)

function Ju(s, T1, T2, P::LinPro, x, phim = expm(-P.B*(T2-T1)*(1. - (s-T1)/(T2-T1))^2))
     sl = P.lambda*(T2-T1)/(T2-s)^2
    ( phim*sl*phim'-sl)\x
end

function Ju(s, T1, T2, P::Ptilde, x)
    H(tofs(s, T1, T2), T2, P, x)*(T2-s)^2/(T2-T1)
end
function J(s, T1, T2, P::Ptilde)
    H(tofs(s, T1, T2), T2, P)*(T2-s)^2/(T2-T1)
end

ubridge(W, Po) = ubridge!(copy(W), W, Po)
function ubridge!{T}(X, W::SamplePath{T}, Po)
    T1 = Po.t0
    T2 = Po.t1
    v = Po.v1
    Pt = ptilde(Po)
    N = length(W)
    N != length(X) && error("X and W differ in length.")
    
    ss = W.tt   
    ww = W.yy
    tt = X.tt
    xx = X.yy
   
    

    u = uofx(T1, Po.v0, T1, T2, v, Pt)
   
    for i in 1:N-1
        s = ss[i]
        t, x = txofsu(s, u, T1, T2, v, Pt)
        tt[i], xx[.., i] = t, x
        bU = 2/(T2-T1)*dotVs(s, T1, T2, v, Pt) - 2/(T2-T1)*b(t, x, Po.Target) +   1/(T2-s)*(u - 2.*a(t, x, Po)*Ju(s, T1, T2, Pt, u) )
        σUdW = (-sqrt(2.0/((T2-T1)*(T2-s))))*(σ(t, x, Po)*(ww[.., i+1]-ww[..,i]))
        u += bU*(ss[i+1]-s) + σUdW
    end
    xx[.., N] = v
    X
end

uthetamethod(W, P, theta=0.5) = uthetamethod!(copy(W), W, Po, theta)
function uthetamethod!(Y, u, W::SamplePath, Po, theta=0.5)

    T1 = Po.t0
    T2 = Po.t1
    v = Po.v1
    Pt = ptilde(Po)
    N = length(W)
    N != length(X) && error("X and W differ in length.")
  
    assert(constdiff(P))


    ss = W.tt   
    ww = W.yy
    tt = X.tt
    xx = X.yy

    u = uofx(T1, Po.v0, T1, T2, v, Pt)

    for i in 1:N-2 # fix me
        s = ss[i]
        t, x = txofsu(s, u, T1, T2, v, Pt)
        tt[i], xx[.., i] = t, x
        x2 = x
        dw = (ww[.., i+1]-ww[..,i])
        delta1 = 2/(T2-T1)*dotVs(s, T1, T2, v, Pt) - 2/(T2-T1)*b(t, x, Po.Target) +   1/(T2-s)*(u - 2.*a(t, x, Po)*Ju(s, T1, T2, Pt, u) )*(ss[i+1]-s) 
        local delta2
       
        const eps2 = 5e-6
        const MM = 8
        for mm in 1:MM
            
            delta2 = 2/(T2-T1)*dotVs(s, T1, T2, v, Pt) - 2/(T2-T1)*b(t, x2, Po.Target) +   1/(T2-s)*(u2 - 2.*a(t, x2, Po)*Ju(s, T1, T2, Pt, u2) )*(ss[i+1]-s) 
            buderiv = -2/(T2-T1)*bderiv(t, x2, Po.Target) + 1/(T2-s)*(I - 2.*a(t, x, Po)*J(s, T1, T2, Pt))*(ss[i+1]-s) 
            bderiv(tt[i+1], y2, P)
            dy2 = -inv(I - theta*(buderiv*(s[i+1]-s)))*(u2 - u - (1-theta)*delta1 - theta*delta2 - σ(tt[i], y, P)*dw)
            
            x2 += dx2
            x2 = txofsu(s, u2, T1, T2, v, Pt)
            if  maximum(abs(dx2)) < eps2
                break;
            end
            
            if mm == MM
                warn("thetamethod: no convergence $i $y $y2  $dy2")
            end
        end
        u = u + (1-theta)*delta1 + theta*delta2 + σ(tt[i], x, P)*dw
    end
    yy[.., N-1] = y
    Y
end

# using left approximation
function ullikelihood{T}(Y::SamplePath{T}, Po)
    yy = Y.yy
    tt = Y.tt
    T1 = Po.t0
    T2 = Po.t1
    v = Po.v1
    P = Po.Target
    Pt = ptilde(Po)
    s2 = soft(tt[1], T1, T2)
    som::Float64 = 0.
    for i in 1:length(tt)-1 #skip last value, summing over n-1 elements
        t = tt[i]
        x = yy[.., i]
        s = s2
        s2 = soft(tt[i+1], T1, T2)
      
        j = J(s, T1, T2, Pt)
        ju = j*uofx(s, yy[.., i], T1, T2, v, Pt) 
        som += 2.*dot(b(t, x, P)  - b(t,x, Pt),ju)*(s2-s)
        
        if !constdiff(Po)
            ad = a(t,x, P) - a(t,x, Pt)
            som += -1./(T2-s)*(trace(j*ad) - T*dot(ju,ad*ju))*(s2-s)
        end
    end
    som
end
function ullikelihoodtrapez{T}(Y::SamplePath{T}, Po)
    yy = Y.yy
    tt = Y.tt
    T1 = Po.t0
    T2 = Po.t1
    v = Po.v1
    P = Po.Target
    Pt = ptilde(Po)
    ss = soft(tt, T1, T2)
    som::Float64 = 0.
    for i in [1]
        t = tt[i]
        x = yy[.., i]
        j = J(ss[i], T1, T2, Pt)
        ju = j*uofx(ss[i], yy[.., i], T1, T2, v, Pt) 
        som += dot(b(t, x, P)  - b(t,x, Pt),ju)*(ss[2]-ss[1])
    end
    
    for i in 2:length(tt)-1 #skip last value, summing over n-1 elements
        t = tt[i]
        x = yy[.., i]
        j = J(ss[i], T1, T2, Pt)
        ju = j*uofx(ss[i], yy[.., i], T1, T2, v, Pt) 
        som += dot(b(t, x, P)  - b(t,x, Pt),ju)*(ss[i+1]-ss[i-1])
        
        if !constdiff(Po)
            error("not implemented")
        end
    end
    som
end

uinnovations(Y, Po) = uinnovations!(copy(Y), Y, Po)
function uinnovations!{T}(W, Y::SamplePath{T}, Po)

    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    yy = Y.yy
    tt = Y.tt
    ww = W.yy
    ss = W.tt
    Pt = ptilde(Po)
    
    T1 = Po.t0
    T2 = Po.t1
    v = Po.v1

    w = zero(ww[.., 1])
    s = s2 = soft(tt[1], T1, T2)
    u2 = uofx(s2, yy[.., 1], T1, T2, v, Pt) 
        
    for i in 1:N-1
        t = tt[i]
        s, u = s2, u2
        ww[.., i] = w
        ss[i] = s

        s2 = soft(tt[i+1], T1, T2)
        u2 = uofx(s2, yy[.., i+1], T1, T2, v, Pt) 
        
        
        bU = 2/(T2-T1)*dotVs(s, T1, T2, v, Pt) - 2/(T2-T1)*b(t,  yy[.., i], Po.Target) +   1/(T2-s)*(u - 2.*a(t,  yy[.., i], Po)*Ju(s, T1, T2, Pt, u) )
        σU = -sqrt(2.0/((T2-T1)*(T2-s)))*σ(t, yy[.., i], Po)
        
        w = w + inv(σU)*(u2 - u - bU*(s2 - s)) 
    end
    ww[.., N] = ww[.., N-1] + randn(typeof(ww[.., N]))*sqrt(s2 - s)
    SamplePath{T}(ss, ww)
end
