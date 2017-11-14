
#module Visualize
#using Bridge, PyPlot, StaticVector
#end

using Bridge, StaticArrays, Bridge.Models
const R = ℝ
srand(1)

iterations = 5000
rho = 0.25 # 
independent = false # true independent proposals
t = 1.0
T = 5.00
n = 10001 # total imputed length
m = 100 # number of segments
M = div(n-1,m)
skiplast = 0
skippoints = 1
dt = (T-t)/(n-1)
tt = t:dt:T
npath = 1
si = 3.
P = Bridge.Models.Lorenz(ℝ{3}(10, 20, 8/3), ℝ{3}(si,si,si))


x0 = Models.x0(P)
#x0 =  ℝ{3}(6, 0, 2)
    
crit(θ1, θ3) = θ1*(θ1 + θ3 + 3)/(θ1 - θ3 - 1)

W = sample(tt, Wiener{ℝ{3}}())
X = SamplePath(tt, zeros(ℝ{3}, length(tt)))
Bridge.solve!(Euler(), X, x0, W, P)
W = sample(tt, Wiener{ℝ{3}}())
X2 = SamplePath(tt, zeros(ℝ{3}, length(tt)))
Bridge.solve!(Euler(), X2, x0, W, P)


# Observation scheme and subsample
_pairs(collection) = Base.Generator(=>, keys(collection), values(collection))

L = I
Σ = 0.5I
lΣ = chol(Σ)'
RV = ℝ{3}

V = SamplePath(collect(_pairs(X))[1:M:end])
#Vo = copy(V)
map!(y -> L*y + lΣ*randn(RV), V.yy, V.yy)

XX = Vector{typeof(X)}(m)
XXmean = Vector{typeof(X)}(m)
XXᵒ = Vector{typeof(X)}(m)
WW = Vector{typeof(W)}(m)
WWᵒ = Vector{typeof(W)}(m)



Xtrue = copy(X)


#=
GP = Bridge.GuidedBridge(tt, X.yy[1], X.yy[end], P, Pt)
V = SamplePath(tt[9500:end], zeros(R{3}, n))
solvebackward!(Bridge.R3(), Bridge.b, V, X.yy[end], P)
V = SamplePath(tt, zeros(R{3}, n))
solve!(Bridge.R3(), Bridge.b, V, X.yy[1], P)
=#




# Create linear noise approximations

forward = false # direction of the linear noise approximation

TPt = Bridge.LinearNoiseAppr{Bridge.Models.Lorenz,StaticArrays.SDiagonal{3,Float64},SVector{3,Float64}}
TPᵒ = Bridge.GuidedBridge{SVector{3,Float64},StaticArrays.SArray{Tuple{3,3},Float64,2,9},Bridge.LinearNoiseAppr{Bridge.Models.Lorenz,StaticArrays.SDiagonal{3,Float64},SVector{3,Float64}},Bridge.Models.Lorenz}


Pt = Vector{TPt}(m)
Pᵒ = Vector{TPᵒ}(m)
H♢ = Bridge.outer(zero(x0))
#v = Xtrue.yy[end]
v = ( L' * inv(Σ) * L)\(L' * inv(Σ) *  V.yy[end])
H♢ = one(Bridge.outer(zero(x0)))*inv( L' * inv(Σ) * L)
#zero(ℝ{3})

for i in m:-1:1
    XX[i] = SamplePath(X.tt[1 + (i-1)*M:1 + i*M], X.yy[1 + (i-1)*M:1 + i*M])
    WW[i] = SamplePath(W.tt[1 + (i-1)*M:1 + i*M], W.yy[1 + (i-1)*M:1 + i*M])
    # short-cut, take v later
    # Pt[i] = Bridge.LinearNoiseAppr(XX[i].tt, P, XX[i].yy[end], Bridge.a(XX[i].tt[end], XX[i].yy[end], P), forward)
    Pt[i] = Bridge.LinearNoiseAppr(XX[i].tt, P, v, Bridge.a(XX[i].tt[end], v, P), forward)
    Pᵒ[i] = Bridge.GuidedBridge(XX[i].tt, P, Pt[i], v, H♢)
    H♢, v = Bridge.gpupdate(Pᵒ[i], L, Σ, V.yy[i])
end

y = x0
for i in 1:m
    sample!(WW[i], Wiener{ℝ{3}}())
    y = Bridge.bridge!(XX[i], y, WW[i], Pᵒ[i])
    XXᵒ[i] = copy(XX[i])
    WWᵒ[i] = copy(WW[i])
    XXmean[i] = SamplePath(XX[i].tt, zeros(XX[i].yy))
end

Y = []
mcstate = [mcstart(XX[i].yy) for i in 1:m]
acc = 0
for it in 1:iterations
    y = x0
    for i in 1:m
        sample!(WWᵒ[i], Wiener{ℝ{3}}())
        if !independent
            rho_ = rho * rand()
            WWᵒ[i].yy[:] = sqrt(rho_)*WWᵒ[i].yy + (sqrt(1-rho_))*WW[i].yy
        end
        y = Bridge.bridge!(XXᵒ[i], y, WWᵒ[i], Pᵒ[i])
    end
    push!(Y, y)

    ll = 0.0
    for i in 1:m-1
        ll += llikelihood(LeftRule(), XXᵒ[i],  Pᵒ[i]) - llikelihood(LeftRule(), XX[i],  Pᵒ[i])
    end
    let i = m
        ll += llikelihood(LeftRule(), XXᵒ[i],  Pᵒ[i], skip=skiplast) - llikelihood(LeftRule(), XX[i],  Pᵒ[i], skip=skiplast)
    end
    print("$it ll $(round(ll,2)) ")

    #if true
    if  rand() < exp(ll) 
        acc += 1
        print("\t X")
        for i in 1:m
            XX[i], XXᵒ[i] = XXᵒ[i], XX[i]
            WW[i], WWᵒ[i] = WWᵒ[i], WW[i]
        end
    else 
        print("\t .")
    end
    println()
    for i in 1:m
        #XXmean[i].yy[:] += XX[i].yy
        mcstate[i] = Bridge.mcnext!(mcstate[i],XX[i].yy)
    end
end
XXstd = Vector{Any}(m)
for i in 1:m
    xx, vv = Bridge.mcstats(mcstate[i])
    XXmean[i].yy[:] = xx
    XXstd[i] = map(x->sqrt.(diag(x)), vv)
end
@show acc/iterations    

# Plot result
include("plotsmoothing.jl")