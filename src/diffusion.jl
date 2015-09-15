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