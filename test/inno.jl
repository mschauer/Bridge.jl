using Bridge, Distributions
using PyPlot
using Base.Test

#srand(5)
h = 1e-7    
n, m = 10, 10000
T1 = 1.
T2 = 2.
T = T2-T1
ss = linspace(T1, T2, n)
tt = Bridge.tofs(ss, T1, T2)

u = 0.5
v = 0.3
a = .7
a2 = 0.4
P = LinPro(-0.8, 0., sqrt(a))
P2 = LinPro(-0.8, 0., sqrt(a2))

la = 1
cs = Bridge.CSpline(T1, T2, la*P.B*u, la*P.B*v)
Po = BridgeProp(P, T1, u, T2, v, a, cs)
Pt = Bridge.ptilde(Po)

cs2 = Bridge.CSpline(T1, T2, P2.B*u, P2.B*v)
Po2 = BridgeProp(P2, T1, u, T2, v, a2, cs)
Pt2 = Bridge.ptilde(Po2)



s = ss[div(n,2)]
t = tt[div(n,2)]
 
 

p = pdf(transitionprob(T1, u, T2, P), v)
pt = exp(lptilde(Po))

C = []
Co = []
Cnames = []

push!(Cnames, "TCSGP + Inno + Update")
z = Float64[
                  begin
                      X = ubridge(sample(ss, Wiener{Float64}()), Po2)
                      Z = Bridge.uinnovations(X, Po2) 
                      Z2 = sample(Z.tt, Wiener{Float64}())
                      Z.yy[:] = sqrt(.8)*Z.yy + sqrt(0.2)*Z2.yy
                      Z.yy[end]
                      X = ubridge(Z, Po)
                      Bridge.llikelihoodleft(X, Po)
                  end
                  for i in 1:m]

o = mean(exp(z)*pt/p); push!(Co, o); push!(C, abs(o - 1)*sqrt(m)/std(exp(z)*pt/p))
display([Cnames C Co])

X = ubridge(sample(ss, Wiener{Float64}()), Po)
K = 100000
σ = sqrt(a2)
θ = 0.1

scaleσ = 0.05
scaleθ = 0.05
param(θ, σ) = -θ, 0., σ^2

proptype = :BP
function MyProp(u, v, P, proptype)
    prev = 1+(abs(P.B/10))
    BridgeProp(P, T1, u, T2, v, Bridge.a(T,v, P), Bridge.CSpline(T1, T2, P.B*prev*u + prev, P.B*prev*v + prev))
end
θsum = σsum = 0.
using Bridge: ptilde, soft, J, uofx, b, constdiff
pacc = bacc = 0

u0 = 0.5
u = 1.
v = .8
P =  LinPro(param(θ, σ)...)
P =  MyProp(u, v, P, proptype)
Z = sample(ss, Wiener{Float64}())
X = ubridge(Z, P)

Y = X
Po = P
MU = 1.
for k in 1:K
    
    if k % 500 == 0
        show(plot(X.tt, X.yy))
    end
    P = LinPro(param(θ, σ)...)
    P° = MyProp(u, v, P, proptype)
    Z° = sample(ss, Wiener{Float64}())
    #Z°.yy[:] = sqrt(0.5)*Z.yy + sqrt(0.5)*Z°.yy 
    X = ubridge(Z, P°)
    X° = ubridge(Z°, P°)


    llold =  Bridge.ullikelihoodtrapez(X, P°)
    llnew =  Bridge.ullikelihoodtrapez(X°, P°)

    if rand() < exp(MU*(llnew - llold))
        bacc += 1
        X = X°
        Z = Z°
    end


    

    σ° = σ .* exp(scaleσ .* randn()) 
    σ° = max(0.1, σ°)
    θ° = θ + (2rand() .- 1).*scaleθ
    Pσ = LinPro(param(θ, σ)...)
    Pσ° = LinPro(param(θ°, σ°)...)
    P° = MyProp(u, v, Pσ, proptype)
    P°° = MyProp(u, v, Pσ°, proptype)

    #Z = Bridge.uinnovations(X, P°)
    X° = Bridge.ubridge(Z, P°°)
 
 

    Y = X°
    Po = P°°

    ll = lp(T1, u0, T2, u, Pσ°) - lp(T1, u0, T2, u, Pσ) +  lptilde(P°°) - lptilde(P°) + Bridge.ullikelihood(X°, P°°) - Bridge.ullikelihood(X, P°)
#    ll = lp(T1, u0, T2, u, Pσ°) - lp(T1, u0, T2, u, Pσ) + lp(T1, u, T2, v, Pσ°) - lp(T1, u, T2, v, Pσ)
    if ll > 1e12 || σ° + 1/σ° > 1e6
        error("Trouble")
    end



    acc = rand() < exp(MU*ll)
    if acc #* (piσ²(σ°.^2)/piσ²(σ.^2)) * (prod(σ°.^2)/prod(σ.^2))
            pacc += 1
            σ = σ°
            θ = θ°
            X = X°
#            Z = Z°
    end
    σsum += σ
    θsum += θ
    println(k, " ", round([θsum/k; θ; σsum/k; σ; ll; bacc/k; pacc/k], 3), "\t [1.5, 1.2]")
    #println(k, " ", round([θ; σ; ll; bacc/k; pacc/k], 3))
end
