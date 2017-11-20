
#module Visualize
#using Bridge, PyPlot, StaticVector
#end

using Bridge, StaticArrays, Bridge.Models
const R = ℝ
srand(2)

iterations = 15000
rho = 0.05 # 
independent = false # true independent proposals
adaptive = true # adaptive smoothing
t = 1.0
T = 5.00
n = 50001 # total imputed length
m = 100 # number of segments
M = div(n-1,m)
skiplast = 0
skippoints = 2
dt = (T-t)/(n-1)
tt = t:dt:T
si = 3.
P = Bridge.Models.Lorenz(ℝ{3}(10, 20, 8/3), ℝ{3}(si,si,si))
P2 = Psmooth = Bridge.Models.Lorenz(ℝ{3}(10, 28, 8/3), ℝ{3}(0,0,0))


x0 = Models.x0(P)
#x0 =  ℝ{3}(6, 0, 2)

W = sample(tt, Wiener{ℝ{3}}())
X = SamplePath(tt, zeros(ℝ{3}, length(tt)))
Bridge.solve!(Euler(), X, x0, W, P)
W = sample(tt, Wiener{ℝ{3}}())
X2 = SamplePath(tt, zeros(ℝ{3}, length(tt)))
Bridge.solve!(Euler(), X2, x0, W, P2)
X2.yy[:] *= 0.7

Xtrue = copy(X)

# Observation scheme and subsample
_pairs(collection) = Base.Generator(=>, keys(collection), values(collection))

L = I
Σ = SDiagonal(50., 1., 1.)
lΣ = chol(Σ)'
RV = ℝ{3}

V = SamplePath(collect(_pairs(Xtrue))[1:M:end])
#Vo = copy(V)
map!(y -> L*y + lΣ*randn(RV), V.yy, V.yy)

XX = Vector{typeof(X)}(m)
XXmean = Vector{typeof(X)}(m)
XXᵒ = Vector{typeof(X)}(m)
WW = Vector{typeof(W)}(m)
WWᵒ = Vector{typeof(W)}(m)


# Create linear noise approximations

forward = false # direction of the linear noise approximation

TPt = Bridge.LinearNoiseAppr{Bridge.Models.Lorenz,StaticArrays.SDiagonal{3,Float64},SVector{3,Float64}}
TPᵒ = Bridge.GuidedBridge{SVector{3,Float64},StaticArrays.SArray{Tuple{3,3},Float64,2,9},Bridge.LinearNoiseAppr{Bridge.Models.Lorenz,StaticArrays.SDiagonal{3,Float64},SVector{3,Float64}},Bridge.Models.Lorenz}


Pt = Vector{TPt}(m)
Pᵒ = Vector{TPᵒ}(m)
#H♢ = Bridge.outer(zero(x0))
#v = Xtrue.yy[end]
v = ( L' * inv(Σ) * L)\(L' * inv(Σ) *  V.yy[end])
H♢ = one(Bridge.outer(zero(x0)))*inv( L' * inv(Σ) * L)

for i in m:-1:1
    XX[i] = SamplePath(X.tt[1 + (i-1)*M:1 + i*M], X.yy[1 + (i-1)*M:1 + i*M])
    WW[i] = SamplePath(W.tt[1 + (i-1)*M:1 + i*M], W.yy[1 + (i-1)*M:1 + i*M])
    # short-cut, take v later
    #Pt[i] = Bridge.LinearNoiseAppr(XX[i].tt, P, XX[i].yy[end], Bridge.a(XX[i].tt[end], XX[i].yy[end], P), forward)#
    Pt[i] = Bridge.LinearNoiseAppr(XX[i].tt, P, v, Bridge.a(XX[i].tt[end], v, P), forward)
    Pt[i].Y.yy[:] *= 0    
    Pᵒ[i] = Bridge.GuidedBridge(XX[i].tt, P, Pt[i], v, H♢)
    H♢, v = Bridge.gpupdate(Pᵒ[i], L, Σ, V.yy[i])
end
 

y = x0
for i in 1:m
    sample!(WW[i], Wiener{ℝ{3}}())
    y = Bridge.bridge!(XX[i], y, WW[i], Pᵒ[i])
end
XXmean = [zero(XX[i]) for i in 1:m]

#v0 = V.yy[1]
#v0 = v
#μ0 = ℝ{3}(0,0,0)
#Σ0 = SDiagonal(40., 40., 40.)
#π0 = Bridge.Gaussian(μ0 + Σ0*L'*inv(L*Σ0*L' + Σ)*(v0 - L*μ0), Σ0 - Σ0*L'*inv(L*Σ0*L' + Σ)*L*Σ0)

#H♢, v = Bridge.gpupdate(Pᵒ[1], L, Σ0, V.yy[1])
π0 = Bridge.Gaussian(v, H♢)
X0 = ℝ{3}[]
function smooth(π0, XX, WW, P, Pᵒ, iterations, rho; verbose = true, independent = false, skiplast = 0)
    m = length(XX)
    rho0 = rho / 2 
    # create workspace
    XXᵒ = deepcopy(XX)
    WWᵒ = deepcopy(WW)

    # initialize
    mcstate = [mcstart(XX[i].yy) for i in 1:m]
    acc = 0
    y0 = π0.μ

    for it in 1:iterations

        if adaptive && it % 2500 == 0 # adaptive smoothing
            v = ( L' * inv(Σ) * L)\(L' * inv(Σ) *  V.yy[end])
            H♢ = one(Bridge.outer(zero(x0)))*inv( L' * inv(Σ) * L)
            for i in m:-1:1
                xx = mcstate[i][1]
                Pt[i].Y.yy[:] = [mean(xx[max(1, j-10):min(end, j+10)]) for j in 1:length(xx)]
                Pᵒ[i] = Bridge.GuidedBridge(XX[i].tt, P, Pt[i], v, H♢)
                H♢, v = Bridge.gpupdate(Pᵒ[i], L, Σ, V.yy[i])
            end
            π0 = Bridge.Gaussian(v, H♢)
        end

        push!(X0, y0)
        if !independent
            y0ᵒ = π0.μ + sqrt(rho0)*(rand(π0) - π0.μ) + sqrt(1-rho0)*(y0 - π0.μ) 
        else
            y0ᵒ = rand(π0) 
        end
        y = y0ᵒ
        for i in 1:m
            sample!(WWᵒ[i], Wiener{ℝ{3}}())
            if !independent
                rho_ = rho * rand()
                WWᵒ[i].yy[:] = sqrt(rho_)*WWᵒ[i].yy + sqrt(1-rho_)*WW[i].yy
            end
            y = Bridge.bridge!(XXᵒ[i], y, WWᵒ[i], Pᵒ[i])
        end


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
            verbose && print("\t X")
            y0 = y0ᵒ
            for i in 1:m
                XX[i], XXᵒ[i] = XXᵒ[i], XX[i]
                WW[i], WWᵒ[i] = WWᵒ[i], WW[i]
            end
        else 
            verbose && print("\t .")
        end
        println("\t\t", round.(y0, 2))
        for i in 1:m
            mcstate[i] = Bridge.mcnext!(mcstate[i],XX[i].yy)
        end
    end
    mcstate, acc
end

mcstate, acc = smooth(π0, XX, WW, P, Pᵒ, iterations, rho; verbose = true, independent = false, skiplast = 0)

XXstd = Vector{Any}(m)
XXrot = Vector{Any}(m)
XXscal = Vector{Any}(m)

for i in 1:m
    xx, vv = Bridge.mcstats(mcstate[i])
    XXmean[i].yy[:] = xx
    XXstd[i] = map(x->sqrt.(diag(x)), vv)
    #XXchol[i] = chol.(Hermitian.(vv))
    XXscal[i] = map(x->sqrt.(svd(x)[2]), vv)
    XXrot[i] = map(x->(Bridge.quaternion(svd(x)[1])), vv)  
end
@show acc/iterations  
V0 = cov(Bridge.mat(X0[end÷2:end]),2)  
#XXmean[1].yy[1] = mean(Bridge.mat(X0[end÷2:end]),2)
#XXscal[1][1] = 2sqrt.(svd(V0)[2])
#XXrot[1][1] = (Bridge.quaternion(svd(V0)[1]))

# Plot result
include("makie.jl")