using Test, Bridge

# test the tricky bridge sampling when endpoint is or is not in grid
#srand(10)
G =  GammaProcess(10.0, 1.5)
GB = GammaBridge(1.0, 2.0, G)
n = 1000
@test abs(mean(sample([0.0, 1.0, 3.0],  G).yy[end] for i in 1:1000) -  mean(Bridge.increment(3, G))) < 4*std(Bridge.increment(0.5, G))/sqrt(n)

X1 = sample([0.0, 0.5, 1.0], GB, 0.2)
X2 = sample([0.0, 0.5, 2.0], GB, 0.2)
X3 = sample([0.0, 0.5], GB, 0.2)

@test  X1.tt ≈ [0,1,2]/2
@test  X2.tt ≈ [0,1,4]/2
@test  X3.tt ≈ [0,1]/2

@test  X1.yy[1] == X1.yy[1] == X1.yy[1] == 0.2

@test  X1.yy[3] ≈ 2.0
@test  X2.yy[3] >= 2.0
@test  X3.yy[2] <= 2.0

γᵒ, γ = 0.75, 1.2
t = 10000.
α = 16.0
n = 10000
tt = range(0, stop=t, length=n)
P = GammaProcess(γ, α)
X = sample(tt, P)

Y = copy(X)
Bridge.uniform_thinning!(Y, P, γᵒ)

@test abs(γ*t/n/α - mean(diff(X.yy))) < 0.2/sqrt(n)
@test abs(γᵒ*t/n/α - mean(diff(Y.yy))) < 0.2/sqrt(n)

γ = 10.
α = 5.
P = GammaProcess(γ, α)
T = 5.0
tt = 0.0:T/n:T
m = 1000
sigma = (t, x = 0.0) -> exp(-cos(t))
L = SamplePath(tt, sample(tt, P).yy .- sample(tt, P).yy)
X = solve(EulerMaruyama(), 0.0, L, ((t,x)-> -0*x, sigma))