"""
    tofs(s, tmin, T)
    soft(t, tmin, T)
  
    Time change mapping t in [t_1, t_2] (X-time) to s in [0, T=t_2 - t_1] (U-time).
"""      
tofs(s, tmin, T) = tmin .+ s.*(2. .- s/T) 

"""
    soft(t, tmin, T)
  
    Time change mapping s in [0, T=t_2 - t_1] (U-time) to t in [t_1, t_2] (X-time), and inverse.
"""  
soft(t, tmin, T) = T-sqrt(T*(T + tmin - t))
#t = tmin + s*(2 - s/T) = tmin + T - T(1 - s/T)^2 = tmax - (T - s)^2/T

euleru(u, W, P) = euleru!(copy(W), u, W, P)
xofu(s, u, T, v, P) = Vs(s, T, v, P) .- (T-s)*u
uofx(s, x, T, v, P) = (Vs(s, T, v, P) .- x)/(T-s)
txofsu(s, u, tmin, T, v, P) = (tofs(s, tmin, T), xofu(s, u, T, v, P))

"""
    Vs (s, T, v, B, beta)

Time changed V for generation of U
"""
function Vs(s, T, v, P::LinPro, phim = expm(-P.B*T*(1. - s/T)^2))
    phim*( v .+ P.betabyB) .-  P.betabyB
end

"""
    dotVs (s, T, v, B, beta)

Time changed time derivative of V for generation of U
"""
function dotVs(s, T, v, P::LinPro, phim = expm(-P.B*T*(1. - s/T)^2))
    phim*( P.B*v .+ P.beta) 
end


function Ju(s,T, P::LinPro, x, phim = expm(-T*(1. - s/T)^2*P.B))
     sl = P.lambda*T/(T-s)^2
    ( phim*sl*phim'-sl)\x
end

euleru(xstart, W, P) = euleru!(copy(W), xstart, W, P)
function euleru!{T}(X, xstart, W::SamplePath{T}, P)
    Pt = P.Pt
    N = length(W)
    N != length(Y) && error("U and W differ in length.")
    
    ww = W.yy
    ss = U.tt
    xx = X.yy
    tt = W.tt
    ss[:] = tt

    u = uofx(0., xstart, T, v, Pt)
    
    for i in 1:N-1
        s = ss[i]
        t, x = txofsu(s, u, tmin, T, v, Pt)
        xx[.., i] = x
        bU = 2/T*dotVs(s,T,v, Pt) - 2/T*b(t, x, P.Target) +   1/(T-s)*(u - 2.*a(t, x, P)*Ju(s, T, Pt, u) )
        σU = -sqrt(2.0/(T*(T-s)))*σ(t, x, P)
        u += bU*(ss[i+1]-s) + σU*(ww[.., i+1]-ww[..,i])
    end
    xx[.., N] = v
    U
end


