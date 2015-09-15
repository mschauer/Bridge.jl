lp(s, x, t, y, P) = logpdf(transitionprob(s,x,t,P),y)

function llikelihood(X::CTPath, P::CTPro)
    ll = 0.
    for i in 2:length(X.tt)
        ll += lp(X.tt[i-1], X.yy[i-1], X.tt[i], X.yy[i], P)
    end
    ll
end


function sample{T}(tt, P::CTPro{T}, x1=zero(T))
    tt = collect(tt)
    yy = zeros(T,length(tt))
    
    yy[1] = x = x1
    for i in 2:length(tt)
        x = rand(transitionprob(tt[i-1], x, tt[i], P))
        yy[i] = x
    end
    CTPath{T}(tt, yy)
end





"""
quvar(X)
             
    Computes quadratic variation of ``X``.
"""
function quvar(X::CTPath)
        sum(diff(X.yy).^2)
end



"""
bracket(X)
bracket(X,Y)
  
     Computes quadratic variation process of ``x`` (of ``x`` and ``y``).
"""     
function bracket(X::CTPath)
        cumsum0(diff(X.yy).^2)
end

function bracket(X::CTPath,Y::CTPath)
        cumsum0(diff(X.yy).*diff(X.yy))
end

"""
ito(Y, X)

    Integrate a valued stochastic process with respect to a stochastic differential.
"""
function ito{T}(X::CTPath, W::CTPath{T})
        assert(X.tt[1] == W.tt[1])
        n = length(X)
        yy = similar(W.yy, n)
        yy[1] = zero(T)
        for i in 2:n
                assert(X.tt[i] == W.tt[i])
                yy[i] = yy[i-1] + X.yy[i-1]*(W.yy[i]-W.yy[i-1])
        end
        CTPath{T}(X.tt,yy) 
end





 