
using Bridge, StaticArrays
using Base.Test

using PyPlot
import PyPlot: plot

include("../example/ellipse.jl")
 
function plot(Y::SamplePath{Float64}; keyargs...) 
    plot(Y.tt, Y.yy; keyargs...)
end    
function plot(Y::SamplePath{SVector{2,Float64}}; keyargs...) 
    yy = Bridge.mat(Y.yy)
    plot(yy[1,:], yy[2,:]; keyargs...)
end

function plot(yy::Vector{SVector{2,Float64}}; keyargs...) 
    yy2 = Bridge.mat(yy)
    plot(yy2[1,:], yy2[2,:]; keyargs...)
end






abstract type LinearProcess{T,S} end

const Vec = SVector{2,Float64}
const Mat = SMatrix{2,2,Float64,4}

struct Linear1 <: LinearProcess{Vec,Mat}

end
struct Linear2 <: LinearProcess{Vec,Mat}

end


β(t, P::Linear1) = 5*Vec(0.2, 0.2)
B(t, P::Linear1) = Mat(-1, 1/2, -1/2, -1)
σ(t, P::Linear1) = Mat(1, 1/10, 0, 1)
P = Linear1()
Pt = LinPro(B(0, P), -B(0, P)\β(0, P), σ(0, P))

β(t, P::Linear2) = Vec(0.2, 0.2)
B(t, P::Linear2) = Mat(-1, cos(t)/2, -cos(t)/2, -1)
σ(t, P::Linear2) = Mat(1, exp(-t)/10, 0, 1)
a(t, P::LinearProcess) = Bridge.outer(σ(t, P))

σ(t, x, P::LinearProcess) = σ(t, P)
a(t, x, P::LinearProcess) = a(t, P)
b(t, x, P::LinearProcess) = B(t, P)*x + β(t, P)

n = 101
T = 5.0
tt = linspace(0, T, n)
dt = T/(n-1)
u = Vec(1.0, 0.3)
v = Vec(.5, -0.3)
mu = zeros(Vec, n)
mu[1] = u
Phi = zeros(Mat, n)
Phi[1] = one(Mat)

for i in 2:length(tt)
    mu[i] = mu[i-1] + (B(tt[i], P)*mu[i-1] + β(tt[i],P))*(tt[i]-tt[i-1])
    Phi[i] = Phi[i-1] + B(tt[i], P)*Phi[i-1]*(tt[i]-tt[i-1])
end

#println(sum(abs.([Phi[i]*u for i in 1:n] - mu)))

 
@test maximum(maximum.(abs.(Phi - [expm(t*Pt.B) for t in tt]))) < T/(2n)
@test maximum([norm(mu[i] - Bridge.mu(0.0, u, tt[i], Pt)) for i in 1:n]) < T/(n)



PhiT = [Phi[n]*inv(Phi[i]) for i in 1:n]


@test maximum(maximum.(abs.(PhiT - [expm((T-t)*Pt.B) for t in tt]))) < T/(2n)
 

K = zeros(Mat, n)

for i in length(tt)-1:-1:1
    K[i] = K[i+1] - Bridge.outer(PhiT[i]*σ(tt[i],P)) *(tt[i]-tt[i+1])
end
@test maximum([norm(K[i]-Bridge.K(tt[i], T, Pt)) for i in 1:n]) < T/n

H = zeros(Mat, n)
V = zeros(Vec, n)
muT = zeros(Vec, n)

V[n] = v
for i in length(tt)-1:-1:1
    dt = (tt[i]-tt[i+1])
    t = (tt[i]+tt[i+1])/2
    H[i] = H[i+1] + (H[i+1]*B(tt[i], P)' +  B(tt[i], P)*H[i+1] -  a(tt[i], P) )*dt
    V[i] = V[i+1] + (B(tt[i], P)*V[i+1] + β(tt[i],P))*dt
    
    #muT[i] = muT[i+1] + PhiT[i]*β(tt[i],P)*dt
    muT[i] = muT[i+1] + expm((T-tt[i])*Pt.B)*β(tt[i],P)*dt
   # muT[i+1] = muT[i+1]  
end

mu0(t, T, mu, B) = expm((T-t)*B)*mu - mu

muT2 = [mu0(t, T, Pt.μ , Pt.B)  for t in tt[1:end]]

@test maximum([norm(muT[i] - mu0(tt[i], T, Pt.μ , Pt.B)  ) for i in 1:n]) < 2T/(n)

#@test maximum([(muT[i] +  expm((T-tt[i])*Pt.B)*v) - Bridge.V(tt[i], T, v, Pt)) for i in 1:n]) < 2T/n


H2 = [abs(T-t) < eps() ? zero(Mat) : inv(Bridge.H(t, T, Pt)) for t in tt[1:end]]
V2 = [Bridge.V(t, T, v, Pt) for t in tt[1:end]]


@test norm(expm((T-T/2)*Pt.B)*(Bridge.V(T/2, T, v, Pt)-u) - (v- Bridge.mu(T/2, u, T, Pt))) < eps(10.0)


@test maximum([(T-tt[i])*norm(inv(H[i]) - Bridge.H(tt[i], T, Pt)) for i in 1:(n-1)]) < 2T/n

@test maximum([norm(V[i] - Bridge.V(tt[i], T, v, Pt))/norm(V[i]) for i in 1:n]) < 2T/n


yy = zeros(Vec,n)
ww = zeros(Vec,n)

N = n
 
H[n] = zero(Mat)
V[n] = v
for i in N-1:-1:1
    dt = (tt[i]-tt[i+1])
    H[i] = H[i+1] + (H[i+1]*B(tt[i], P)' +  B(tt[i], P)*H[i+1] -  a(tt[i], P) ) 
    #V[i] = V[i+1] + (B(tt[i], P)*V[i+1] + β(tt[i],P))*dt
end
y = u
for i in 1:N-1
    yy[.., i] = y
    dt = (tt[i+1]-tt[i])
    y = y + b(tt[i], y, P)*dt + a(tt[i], y, P)*inv(H[i])*(V[i] - y)*dt + σ(tt[i], y, P)*(ww[.., i+1]-ww[..,i])
end
yy[.., N] = v





ii = [searchsortedfirst(tt,t ) for t in linspace(0,T, 20)]
 

 
clf()
 
plot(mu[ii])
axis([-1,1,-1,2]) 
ellipses(mu[ii], K[ii], 1)
plot(yy[ii]) 