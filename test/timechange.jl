using Bridge, FixedSizeArrays, Distributions
using Base.Test

#srand(5)
h = 1e-7    
n, m = 20, 10000
T1 = 1.
T2 = 2.
T = T2-T1
ss = linspace(T1, T2, n)
tt = Bridge.tofs(ss, T1, T2)

u = 0.5
v = 0.3
a = .7
P = LinPro(-0.8, 0., sqrt(a))
cs = Bridge.CSpline(T1, T2, P.B*u, P.B*v)
Po = BridgeProp(P, T1, u, T2, v, a, cs)
Pt = Bridge.ptilde(Po)


s = ss[div(n,2)]
t = tt[div(n,2)]
@test Bridge.V(t, T2, v, Pt) ≈ Bridge.Vs(s, T1, T2, v, Pt)
@test Bridge.dotV(t, T2, v, Pt) ≈ Bridge.dotVs(s, T1, T2, v, Pt)
@test abs((Bridge.V(t+h, T2, v, Pt) - Bridge.V(t, T2, v, Pt))/h - Bridge.dotV(t, T2, v, Pt)) < h

@test tt[1] == T1
@test tt[end] == T2
@test (Bridge.V(T1, T2, v, Pt) - u)/T ≈ Bridge.uofx(T1, Po.v0, T1, T2, v, Pt) # 
@test [T1, u] ≈ [Bridge.txofsu(T1, Bridge.uofx(T1, Po.v0, T1, T2, v, Pt), T1, T2, v, Pt)...]
@test norm(Bridge.soft(Bridge.tofs(1:0.1:2, 1, 2), 1,2 ) .- (1:0.1:2)) < sqrt(eps())

@test norm(Bridge.b(T1, u, P) - Bridge.b(T1, u, Pt)) + norm(Bridge.b(T2, v, P) - Bridge.b(T2, v,Pt)) < sqrt(eps())

X = Bridge.ubridge(sample(ss, Wiener{Float64}()), Po)
@test X.tt ≈ tt


p = pdf(transitionprob(T1, u, T2, P), v)
pt = exp(lptilde(Po))

C = []
z = Float64[
    begin
        X = Bridge.bridge(sample(tt, Wiener{Float64}()), Po)
        llikelihood(X, Po)
    end
    for i in 1:m]
 
push!(C, abs(mean(exp(z)*pt/p-1)*sqrt(m)/std(exp(z)*pt/p)))

z = Float64[
    begin
        X = Bridge.ubridge(sample(ss, Wiener{Float64}()), Po)
        llikelihood(X, Po)
    end
    for i in 1:m]

push!(C, abs(mean(exp(z)*pt/p-1)*sqrt(m)/std(exp(z)*pt/p)))
@test C[end] < 1.96

print(C)

