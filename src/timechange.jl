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
soft(t, T1, T2, T=T2-T1) = T2 - sqrt(T*(T2 - t))


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
        σU = -sqrt(2.0/((T2-T1)*(T2-s)))*σ(t, x, Po)
        u += bU*(ss[i+1]-s) + σU*(ww[.., i+1]-ww[..,i])
    end
    xx[.., N] = v
    X
end


