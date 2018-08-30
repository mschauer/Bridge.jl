
#module Visualize
#using Bridge, PyPlot, StaticVector
#end
using JLD2

simid = 3
sim = [:lorenz1, :lorenz2, :lorenz3][simid]
simname = String(sim)

mkpath(joinpath("output", simname))
try # save cp of this file as documentation
    cp(@__FILE__(), joinpath("output",simname,"$simname.jl"); force=true)
catch
end

iterations = 100000
saveit = 5000
alpha = 0.5 # 1 - exp(-alpha Exp) is AR(1) coefficient of Brownian motion valued random walk  
independent = false # independent proposals
adaptit = 1000 # adapt every `it`th step
adaptmax = iterations

 
if simid == 4
    partial = true
    adaptive = true
    initnu = :backward
elseif simid == 3
    partial = false
    adaptive = true
    initnu = :brown
    alpha = 0.5
elseif simid == 2
    partial = false
    adaptive = false
    initnu = :backward
    alpha = 5.0
elseif simid == 1
    partial = false
    adaptive = false
    initnu = :brown
    alpha = 5.0
else 
    error("provide `simid in 1:3`")
end

include("lorenz.jl")

#M = div(M,10)
#M = M*4

XX = Vector{typeof(X)}(m)
XXmean = Vector{typeof(X)}(m)
XXᵒ = Vector{typeof(X)}(m)
WW = Vector{typeof(W)}(m)
WWᵒ = Vector{typeof(W)}(m)


# Create linear noise approximations

if initnu == :backward
    TPt = Bridge.LinearNoiseAppr{Bridge.Models.Lorenz,StaticArrays.SDiagonal{3,Float64},SVector{3,Float64}}
else
    TPt = Bridge.LinearAppr{StaticArrays.SDiagonal{3,Float64},StaticArrays.SArray{Tuple{3,3},Float64,2,9},SVector{3,Float64}}
end
TPᵒ = Bridge.GuidedBridge{SVector{3,Float64},StaticArrays.SArray{Tuple{3,3},Float64,2,9},TPt,Bridge.Models.Lorenz}

#Pt = Vector{TPt}(m)
#Pᵒ = Vector{TPᵒ}(m)
Pt = Vector(m)
Pᵒ = Vector(m)
H♢, v = Bridge.gpupdate(πH*one(SM), zero(SV), L, Σ, V.yy[end])

for i in m:-1:1
    tt_ = range(V.tt[i], stop=V.tt[i+1], length=M+1) 
    XX[i] = Bridge.samplepath(tt_, zero(SV))
    WW[i] = Bridge.samplepath(tt_, zero(RV))
    
    a_ = Bridge.a(XX[i].tt[end], v, P)

     if initnu == :brown
        Pt[i] = Bridge.linearappr(SamplePath(XX[i].tt, zeros(ℝ{3}, M+1)), Bridge.NoDrift(P))
    elseif initnu == :backward
        Pt[i] = Bridge.LinearNoiseAppr(XX[i].tt, P, v, a_, :backward) 
    elseif initnu == :foci
        if partial
            x_ = Models.foci(P)[1 + ( V.yy[i][1] > 0)]
        else
            x_ = Models.foci(P)[1 + ( V.yy[i][2] + V.yy[i][3] > 0)]
        end
        Pt[i] = Bridge.linearappr(SamplePath(XX[i].tt, fill(x_, M+1)), P)
    end
  

    Pᵒ[i] = Bridge.GuidedBridge(XX[i].tt, P, Pt[i], v, H♢)
    H♢, v = Bridge.gpupdate(Pᵒ[i], L, Σ, V.yy[i])
end
 

π0 = Bridge.Gaussian(v, Hermitian(H♢))

# initialize

y = π0.μ
for i in 1:m
    sample!(WW[i], Wiener{ℝ{3}}())
    y = Bridge.bridge!(XX[i], y, WW[i], Pᵒ[i])
end
XXmean = [zero(XX[i]) for i in 1:m]


function smooth(π0, XX, WW, P, Pᵒ, iterations, alpha; verbose = true,adaptive = true, adaptmax = iterations, adaptit = 5000, saveit = 500, independent = false, 
    smoothmean = false, hwindow=20)
    Paths = []
    
    m = length(XX)

    # create workspace
    XXᵒ = deepcopy(XX)
    WWᵒ = deepcopy(WW)
    Wnr = Wiener{valtype(WW[1])}()

    # initialize
    mcstate = [mcstart(XX[i].yy) for i in 1:m]
    acc = 0
    y0 = π0.μ
    X0 = typeof(y0)[]
    Xt = typeof(y0)[]
    newblock = false
    for it in 1:iterations
        doaccept = false
        if it % saveit == 0
            push!(Paths, collect(Iterators.flatten(XX[i].yy[1:end-1] for i in 1:m)))
        end

        if adaptive && it < adaptmax && it % adaptit == 0 # adaptive smoothing
            H♢, v = Bridge.gpupdate(πH*one(SM), zero(SV), L, Σ, V.yy[end])            
            for i in m:-1:1
                xx = mcstate[i][1]
                if isa(Pᵒ[i].Pt, Bridge.LinearNoiseAppr)
                    if smoothmean
                        Pᵒ[i].Pt.Y.yy[:] = [mean(xx[max(1, j-hwindow):min(end, j+hwindow)]) for j in 1:length(xx)]
                    else
                        Pᵒ[i].Pt.Y.yy[:] = xx
                    end
                elseif isa(Pᵒ[i].Pt, Bridge.LinearAppr)
                    if smoothmean
                        Y = SamplePath(XX[i].tt, [mean(xx[max(1, j-hwindow):min(end, j+hwindow)]) for j in 1:length(xx)])
                    else
                        Y = SamplePath(XX[i].tt, xx)
                    end
                    Bridge.linearappr!(Pᵒ[i].Pt, Y, P)
                else 
                    error("Adaption not implemented for proposals Pᵒ.Pt")
                end
                Pᵒ[i] = Bridge.GuidedBridge(XX[i].tt, P, Pᵒ[i].Pt, v, H♢)
                H♢, v = Bridge.gpupdate(Pᵒ[i], L, Σ, V.yy[i])
            end
            π0 = Bridge.Gaussian(v, Hermitian(H♢))
            #y0ᵒ = rand(π0) 
            newblock = true
            if it == adaptit # first adaptive proposal needs some help sometimes
                doaccept = true
            end

        end

        push!(X0, y0)
        push!(Xt, XX[div(end,2)].yy[1])
        rho_ = exp(-alpha*randexp())   
        if newblock
            #y0ᵒ = rand(π0) 
            y0ᵒ = y0
        elseif !independent
           
            y0ᵒ = π0.μ + sqrt(rho_)*(rand(π0) - π0.μ) + sqrt(1-rho_)*(y0 - π0.μ) 
        else
            y0ᵒ = rand(π0) 
        end
        y = y0ᵒ
        for i in 1:m
            sample!(WWᵒ[i], Wnr)
            if !independent
                WWᵒ[i].yy[:] = sqrt(rho_)*WWᵒ[i].yy + sqrt(1-rho_)*WW[i].yy
            end
            y = Bridge.bridge!(XXᵒ[i], y, WWᵒ[i], Pᵒ[i])
        end


        ll = 0.0
        for i in 1:m
            ll += llikelihood(LeftRule(), XXᵒ[i],  Pᵒ[i]) - llikelihood(LeftRule(), XX[i],  Pᵒ[i])
        end
        verbose && print(@sprintf("%7i", it), " ll ",  @sprintf("%10.3f", ll))

        #if true
        if doaccept || rand() < exp(ll) 
            acc += 1
            verbose && print("\t ✓")
            y0 = y0ᵒ
            for i in 1:m
                XX[i], XXᵒ[i] = XXᵒ[i], XX[i]
                WW[i], WWᵒ[i] = WWᵒ[i], WW[i]
            end
            newblock = false
        else 
            verbose && print("\t .")
        end
        verbose && println([" ", "N"][1 + newblock], "\t", round(acc/it, digits=2), "\t", round.(y0 - x0, 2))
        for i in 1:m
            mcstate[i] = Bridge.mcnext!(mcstate[i],XX[i].yy)
        end
    end
    Paths, X0, Xt, mcstate, acc
end

Paths, X0, Xt, mcstates, acc = smooth(π0, XX, WW, P, Pᵒ, iterations, alpha;
adaptive = adaptive,
adaptit = adaptit,
adaptmax = adaptmax,
saveit = saveit, 
verbose = true, independent = independent)

open(joinpath("output", simname,"info.txt"), "w") do f
    println(f, "acc") 
    println(f, round(acc/iterations, digits=3)) 
end


writedlm(joinpath("output", simname, "x0n$simid.csv"), [1:iterations Bridge.mat(X0)'])
writedlm(joinpath("output", simname, "xtn$simid.csv"), [1:iterations Bridge.mat(Xt)'])



# Plot result
include(joinpath(Pkg.dir("Bridge"),"extra/makie.jl"))

Xmeanm, Xstdm = Bridge.mcmarginalstats(mcstates)


Xmean, Xrot, Xscal = mcsvd3(mcstates)

save(joinpath(ENV["BRIDGE_OUTDIR"],  "$simname$(simid)paths.jld2"), "Path", Paths)
save(joinpath(ENV["BRIDGE_OUTDIR"],  "$simname$(simid)states.jld2"), "mcstates", mcstates)
