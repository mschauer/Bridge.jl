
#module Visualize
#using Bridge, PyPlot, StaticVector
#end

using Bridge, StaticArrays, Bridge.Models
const R = ℝ
Random.seed!(2)

iterations = 25000
rho = 0.005 # 1 - rho is AR(1) coefficient of Brownian motion valued random walk  
independent = false # independent proposals
adaptive = false # adaptive proposals
adaptit = 500 # adapt every `it`th step
#adaptmax = 4000
adaptmax = 10000
cheating = false # take a posteriori good value of Pt
showpath = false
partial = true

πH = 2000. # prior

t = 1.0
T = 5.00
n = 50001 # total imputed length
m = 500 # number of segments
M = div(n-1,m)
skiplast = 0
skippoints = 2
dt = (T-t)/(n-1)
tt = t:dt:T
si = 3.
P = Bridge.Models.Lorenz(ℝ{3}(10, 20, 8/3), ℝ{3}(si,0*si,si))
P2 = Psmooth = Bridge.Models.Lorenz(ℝ{3}(10, 28, 8/3), ℝ{3}(0,0,0))


x0 = Models.x0(P)
#x0 =  ℝ{3}(6, 0, 2)

W = sample(tt, Wiener{ℝ{3}}())
X = SamplePath(tt, zeros(ℝ{3}, length(tt)))
Bridge.solve!(Euler(), X, x0, W, P)
W = sample(tt, Wiener{ℝ{3}}())
X2 = SamplePath(tt, zeros(ℝ{3}, length(tt)))
Bridge.solve!(Euler(), X2, x0, W, P2)


Xtrue = copy(X)

# Observation scheme and subsample
_pairs(collection) = Base.Generator(=>, keys(collection), values(collection))
SV = ℝ{3}
SM = typeof(one(Bridge.outer(zero(SV))))

if !partial
    L = I
    Σ = SDiagonal(1., 1., 1.)
    lΣ = chol(Σ)'
    RV = SV
    RM = SM
 
    V = SamplePath(collect(_pairs(Xtrue))[1:M:end])
    map!(y -> L*y + lΣ*randn(RV), V.yy, V.yy)
else 
    L = @SMatrix [0.0 1.0 0.0; 0.0 0.0 1.0]
    Σ = SDiagonal(1., 1.)
    lΣ = chol(Σ)'
    RV = ℝ{2}
    RM = typeof(one(Bridge.outer(zero(RV))))
 
    V_ = SamplePath(collect(_pairs(Xtrue))[1:M:end])
    V = SamplePath(V_.tt, map(y -> L*y + lΣ*randn(RV), V_.yy))

end
XX = Vector{typeof(X)}(m)
XXmean = Vector{typeof(X)}(m)
XXᵒ = Vector{typeof(X)}(m)
WW = Vector{typeof(W)}(m)
WWᵒ = Vector{typeof(W)}(m)


# Create linear noise approximations


TPt = Bridge.LinearAppr{StaticArrays.SDiagonal{3,Float64},StaticArrays.SArray{Tuple{3,3},Float64,2,9},SVector{3,Float64}}
TPᵒ = Bridge.GuidedBridge{SVector{3,Float64},StaticArrays.SArray{Tuple{3,3},Float64,2,9},TPt,Bridge.Models.Lorenz}


Pt = Vector{TPt}(m)
Pᵒ = Vector{TPᵒ}(m)

H♢, v = Bridge.gpupdate(πH*one(SM), zero(SV), L, Σ, V.yy[end])

for i in m:-1:1
    XX[i] = SamplePath(X.tt[1 + (i-1)*M:1 + i*M], X.yy[1 + (i-1)*M:1 + i*M])
    WW[i] = SamplePath(W.tt[1 + (i-1)*M:1 + i*M], W.yy[1 + (i-1)*M:1 + i*M])
    
      
#    Σ_= Bridge.σ(XX[i].tt[end], v, P)     

    x_ = Models.foci(P)[1 + ( V.yy[i][1] > 0)]
#    B_ = -Bridge.bderiv(0, x_, P)'
#    β_ = zero(ℝ{3})  #-B_*f
#
    #Pt[i] = Bridge.LinearAppr(XX[i].tt, fill(x_, M+1), fill(B_, M+1), fill(β_, M+1), fill(Σ_, M+1)) 

    Pt[i] = Bridge.linearappr(SamplePath(XX[i].tt, fill(x_, M+1)), P)
    
    Pᵒ[i] = Bridge.GuidedBridge(XX[i].tt, P, Pt[i], v, H♢)

  

    H♢, v = Bridge.gpupdate(Pᵒ[i], L, Σ, V.yy[i])
end
 

π0 = Bridge.Gaussian(v, Hermitian(H♢))

y = π0.μ
for i in 1:m
    sample!(WW[i], Wiener{ℝ{3}}())
    y = Bridge.bridge!(XX[i], y, WW[i], Pᵒ[i])
end
XXmean = [zero(XX[i]) for i in 1:m]

#X0 = ℝ{3}[]

function smooth(π0, XX, WW, P, Pᵒ, iterations, rho; verbose = true, adaptive = true, adaptmax = iterations, adaptit = 5000, independent = false, hwindow=20)
    m = length(XX)
    rho0 = rho 
    # create workspace
    XXᵒ = deepcopy(XX)
    WWᵒ = deepcopy(WW)
    Wnr = Wiener{valtype(WW[1])}()

    # initialize
    mcstate = [mcstart(XX[i].yy) for i in 1:m]
    acc = 0
    y0 = π0.μ
    
    for it in 1:iterations
        doaccept = false

        if adaptive && it < adaptmax && it % adaptit == 0 # adaptive smoothing
            H♢, v = Bridge.gpupdate(πH*one(SM), zero(SV), L, Σ, V.yy[end])            
            for i in m:-1:1
                xx = mcstate[i][1]
                Y = SamplePath(XX[i].tt, [mean(xx[max(1, j-hwindow):min(end, j+hwindow)]) for j in 1:length(xx)])
                Bridge.linearappr!(Pt[i], Y, P)
                
                Pᵒ[i] = Bridge.GuidedBridge(XX[i].tt, P, Pt[i], v, H♢)
                H♢, v = Bridge.gpupdate(Pᵒ[i], L, Σ, V.yy[i])
            end
            π0 = Bridge.Gaussian(v, Hermitian(H♢))
            #y0ᵒ = rand(π0) 
            #doaccept = true
        end

        if !independent
            y0ᵒ = π0.μ + sqrt(rho0)*(rand(π0) - π0.μ) + sqrt(1-rho0)*(y0 - π0.μ) 
        else
            y0ᵒ = rand(π0) 
        end
        y = y0ᵒ
        for i in 1:m
            sample!(WWᵒ[i], Wnr)
            if !independent
                rho_ = rho * rand()
                WWᵒ[i].yy[:] = sqrt(rho_)*WWᵒ[i].yy + sqrt(1-rho_)*WW[i].yy
            end
            y = Bridge.bridge!(XXᵒ[i], y, WWᵒ[i], Pᵒ[i])
        end


        ll = 0.0
        for i in 1:m
            ll += llikelihood(LeftRule(), XXᵒ[i],  Pᵒ[i]) - llikelihood(LeftRule(), XX[i],  Pᵒ[i])
        end
        verbose && print("$it ll $(round(ll, digits=2)) ")

        #if true
        if doaccept || rand() < exp(ll) 
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
        verbose && println("\t\t", round.(y0 - x0, 2))
        for i in 1:m
            mcstate[i] = Bridge.mcnext!(mcstate[i],XX[i].yy)
        end
    end
    mcstate, acc
end

if showpath
    using Makie
    scene = Scene(resolution = (400, 400))
    lines((@view Xtrue.yy[1:10:end]), linewidth = 0.2, color = :darkblue)
    center!(scene)
    
    for i in indices(XX,1)
        lines(XX[i].yy, linewidth = 0.2, color = :darkgreen)
    end
end
    


mcstates, acc = smooth(π0, XX, WW, P, Pᵒ, iterations, rho;
    adaptive = adaptive,
    adaptit = adaptit,
    adaptmax = adaptmax,
    verbose = true, independent = independent)
@show acc/iterations  

include("../extra/makie.jl")

Xmean, Xrot, Xscal = mcsvd3(mcstates)

# Plot result
include("makie.jl")
Xmean, Xstd = mcmarginalstats(mcstates)

if true
    using PyPlot
    clf()
    for i in 1:3; plot(Xtrue.tt, getindex.(Xtrue.yy, i), label = "$i", color = :red); end
    for i in 1:3; plot(Xtrue.tt, getindex.(Xmean, i), label = "$i", color = :blue); end
    for i in 1:3; plot(Xtrue.tt, getindex.(Xmean + Xstd, i), color = :lightblue); end
    for i in 1:3; plot(Xtrue.tt, getindex.(Xmean - Xstd, i),   color = :lightblue); end
    for i in 1:2; plot(V.tt, getindex.(V.yy,i), ".",  color = :black); end
    for i in 1:3; plot(Xtrue.tt[1:end-1], getindex.(Nu, i), color = :palevioletred); end
end