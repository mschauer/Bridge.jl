using Bridge
using Base.Test

brown1(s, t, n) = sample(linspace(s, t, n), Wiener{Float64}())
 

function f(n)
    W = brown1(0.,1.,1000)
    X = copy(W)
    s = 0.
    for i in 1:n 
        X.yy .= 0
        s += Bridge.euler!(X, 0.,W, WienerBridge(2., 1.)).yy[end]
    end
    s
end

@time f(1000)
