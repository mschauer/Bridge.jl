
#module Visualize
#using Bridge, PyPlot, StaticVector
#end
using JLD

simid = 2
sim = [:lorenz1, :lorenz2, :lorenz3][simid]
simname = String(sim)

mkpath(joinpath("output", simname))
try # save cp of this file as documentation
    cp(@__FILE__(), joinpath("output",simname,"$simname.jl"); remove_destination=true)
end

iterations = 50000
saveit = 5000
rho = 0.02 # 1 - rho is AR(1) coefficient of Brownian motion valued random walk  
independent = false # independent proposals
adaptive = true # adaptive proposals
adaptit = 1000 # adapt every `it`th step
adaptmax = iterations
  # take a posteriori good value of Pt

if simid == 4
    partial = true
    initnu = :backwards
elseif simid == 3
    partial = false
    adaptive = true
    initnu = :backwards
    #julia> acc/iterations
    #0.11502
elseif simid == 2
    partial = false
    adaptive = false
    initnu = :backwards
    rho = 0.01
elseif simid == 1
    partial = false
    adaptive = false
    initnu = :brown
    #julia> acc/iterations
    #0.08496
    rho = 0.01
else 
    error("provide `simid in 1:3`")
end


include("lorenz.jl")

XX = Vector{typeof(X)}(m)
XXmean = Vector{typeof(X)}(m)
XXᵒ = Vector{typeof(X)}(m)
WW = Vector{typeof(W)}(m)
WWᵒ = Vector{typeof(W)}(m)


# Create linear noise approximations


TPt = Bridge.LinearNoiseAppr{Bridge.Models.Lorenz,StaticArrays.SDiagonal{3,Float64},SVector{3,Float64}}
TPᵒ = Bridge.GuidedBridge{SVector{3,Float64},StaticArrays.SArray{Tuple{3,3},Float64,2,9},Bridge.LinearNoiseAppr{Bridge.Models.Lorenz,StaticArrays.SDiagonal{3,Float64},SVector{3,Float64}},Bridge.Models.Lorenz}


Pt = Vector{TPt}(m)
Pᵒ = Vector{TPᵒ}(m)

H♢, v = Bridge.gpupdate(πH*one(SM), zero(SV), L, Σ, V.yy[end])

for i in m:-1:1
    XX[i] = SamplePath(X.tt[1 + (i-1)*M:1 + i*M], X.yy[1 + (i-1)*M:1 + i*M])
    WW[i] = SamplePath(W.tt[1 + (i-1)*M:1 + i*M], W.yy[1 + (i-1)*M:1 + i*M])
    
    a_ = Bridge.a(XX[i].tt[end], v, P)
    Pt[i] = Bridge.LinearNoiseAppr(XX[i].tt, P, v, a_, :backward) 
    if initnu == :brown
        Pt[i].Y.yy[:] *= 0
    end

    Pᵒ[i] = Bridge.GuidedBridge(XX[i].tt, P, Pt[i], v, H♢)
    H♢, v = Bridge.gpupdate(Pᵒ[i], L, Σ, V.yy[i])
end
 

π0 = Bridge.Gaussian(v, H♢)

y = π0.μ
for i in 1:m
    sample!(WW[i], Wiener{ℝ{3}}())
    y = Bridge.bridge!(XX[i], y, WW[i], Pᵒ[i])
end
XXmean = [zero(XX[i]) for i in 1:m]

#X0 = ℝ{3}[]

function smooth(π0, XX, WW, P, Pᵒ, iterations, rho; verbose = true,adaptive = true, adaptmax = iterations, adaptit = 5000, saveit = 500, independent = false, hwindow=20)
    Paths = []
    
    m = length(XX)
    rho0 = rho / 2 
    # create workspace
    XXᵒ = deepcopy(XX)
    WWᵒ = deepcopy(WW)
    W = Wiener{valtype(WW[1])}()

    # initialize
    mcstate = [mcstart(XX[i].yy) for i in 1:m]
    acc = 0
    y0 = π0.μ
    X0 = typeof(y0)[]
    Xt = typeof(y0)[]
    for it in 1:iterations
        if it % saveit == 0
            push!(Paths, collect(Iterators.flatten(XX[i].yy[1:end-1] for i in 1:m)))
        end

        if adaptive && it < adaptmax && it % adaptit == 0 # adaptive smoothing
            error()
            H♢, v = Bridge.gpupdate(πH*one(SM), zero(SV), L, Σ, V.yy[end])            
            for i in m:-1:1
                xx = mcstate[i][1]
                Pt[i].Y.yy[:] += [mean(xx[max(1, j-hwindow):min(end, j+hwindow)]) for j in 1:length(xx)]
                Pt[i].Y.yy[:] /= 2
                Pᵒ[i] = Bridge.GuidedBridge(XX[i].tt, P, Pt[i], v, H♢)
                H♢, v = Bridge.gpupdate(Pᵒ[i], L, Σ, V.yy[i])
            end
            π0 = Bridge.Gaussian(v, H♢)
        end

        push!(X0, y0)
        push!(Xt, XX[div(end,2)].yy[1])

        if !independent
            y0ᵒ = π0.μ + sqrt(rho0)*(rand(π0) - π0.μ) + sqrt(1-rho0)*(y0 - π0.μ) 
        else
            y0ᵒ = rand(π0) 
        end
        y = y0ᵒ
        for i in 1:m
            sample!(WWᵒ[i], W)
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
        verbose && println("\t\t", round.(y0 - x0, 2))
        for i in 1:m
            mcstate[i] = Bridge.mcnext!(mcstate[i],XX[i].yy)
        end
    end
    Paths, X0, Xt, mcstate, acc
end

Paths, X0, Xt, mcstates, acc = smooth(π0, XX, WW, P, Pᵒ, iterations, rho;
adaptive = adaptive,
adaptit = adaptit,
adaptmax = adaptmax,
saveit = saveit, 
verbose = true, independent = independent)

writecsv(joinpath("output", simname, "x0n$simid.csv"), [1:iterations Bridge.mat(X0)'])
writecsv(joinpath("output", simname, "xtn$simid.csv"), [1:iterations Bridge.mat(Xt)'])
#writecsv(joinpath("output","simid"), )
JLD.save(joinpath("output", simname, "results.jld"), "Path", Paths, "mcstates", mcstates)

#V0 = cov(Bridge.mat(X0[end÷2:end]),2)  

# Plot result
include("../extra/makie.jl")

Xmean, Xrot, Xscal = mcsvd3(mcstates)

include("makie.jl")
