
using Bridge, StaticArrays, Bridge.Models
const R = ℝ
Random.seed!(2)
Random.seed!(3)
πH = 2000. # prior

t = 1.0
T = 5.00
n = 50001 # total imputed length
m = 100 # number of segments
M = div(n-1,m)
skippoints = 2
dt = (T-t)/(n-1)
tt = t:dt:T
si = 3.
# 10, 20, 8/3 srand(2)
θ² = 1.
γ = 1.
P = Bridge.Models.Pendulum(θ², γ)
Psm = Bridge.Models.Pendulum(θ², 0.0)
x0 = Models.x0(P)


W = sample(tt, Wiener())
X = SamplePath(tt, zeros(ℝ{2}, length(tt)))
Bridge.solve!(Euler(), X, x0, W, P)
W = sample(tt, Wiener())
Xsm = SamplePath(tt, zeros(ℝ{2}, length(tt)))
Bridge.solve!(Euler(), Xsm, x0, W, Psm)


Xtrue = copy(X)

# Observation scheme and subsample
_pairs(collection) = Base.Generator(=>, keys(collection), values(collection))
SV = ℝ{2}
SM = typeof(one(Bridge.outer(zero(SV))))
if isdefined(:partial)
    if !partial
        L = I
        Σ = SDiagonal(1., 1.)
        lΣ = chol(Σ)'
        RV = SV
        RM = SM
    
        V = SamplePath(collect(_pairs(Xtrue))[1:M:end])
        map!(y -> L*y + lΣ*randn(RV), V.yy, V.yy)
    else 
        L = @SMatrix [1.0 0.0]
        Σ = isdefined(:Σ_) ? Σ_ : 1.00
        @show Σ
        lΣ = chol(Σ)'
        RV = Float64
        RM = Float64 #typeof(one(Bridge.outer(zero(RV))))
    
        V_ = SamplePath(collect(_pairs(Xtrue))[1:M:end])
        V = SamplePath(V_.tt, map(y -> (L*y)[1] + lΣ*randn(RV), V_.yy))
    end
else 
    println("Info:  No observation scheme defined. Set `partial in [true, false]`")    
end

