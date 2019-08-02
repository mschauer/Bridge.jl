if TEST
    uu = State([Point{ForwardDiff.Dual{Float64}}(2,3) for i in 1:6],  [Point{ForwardDiff.Dual{Float64}}(2,3) for i in 1:6])
    duu .= deepvalue(uu)
    typeof(duu)
end



if false # check later
    function nfs(P::Union{Landmarks,MarslandShardlow})
        if isa(P,Landmarks)
            return(P.nfs)
        end
        if isa(P,MarslandShardlow)
            return(0)
        end
    end

    function dwiener(P::Union{Landmarks,MarslandShardlow})
        if isa(P,Landmarks)
            return(length(P.nfs))
        end
        if isa(P,MarslandShardlow)
            return(P.n)
        end
    end
end




function lm_mcmc(tt_, (LT,ΣT,μT), (L0,Σ0), (xobs0,xobsT), P, Paux, model, sampler, dataset, xinit, δ, ITER,outdir; makefig=true)
    println("compute guiding term:")
    Lt, Mt⁺, μt, Ht = initLMμH(tt_,(LT,ΣT,μT))
    Q = construct_guidedproposal!(tt_, (Lt, Mt⁺ , μt, Ht), (LT,ΣT,μT), (L0, Σ0), (xobs0, xobsT), P, Paux)

    println("Sample guided proposal:")
    X = initSamplePath(tt_, xinit)
    W = initSamplePath(tt_,  zeros(StateW, dwiener))
    sample!(W, Wiener{Vector{StateW}}())

    ll = simguidedlm_llikelihood!(LeftRule(), X, xinit, W, Q; skip=sk)
    if makefig
        if isa(P,Landmarks)
            plotlandmarkpositions(X,P.n,model,xobs0,xobsT,P.nfs;db=4)
        end
        if isa(P,MarslandShardlow)
            plotlandmarkpositions(X,P.n,model,xobs0,xobsT,0;db=4)
        end
    end

    # saving objects
    objvals =   Float64[]  # keep track of (sgd approximation of the) loglikelihood
    acc = zeros(2) # keep track of mcmc accept probs (first comp is for CN update; 2nd component for langevin update on initial momenta)
    #Xsave = zeros(length(subsamples), length(tt_) * P.n * 2 * d )
    Xsave = typeof(zeros(length(tt_) * P.n * 2 * d))[]
    push!(Xsave, convert_samplepath(X))
    push!(objvals, ll)

    mask = deepvec(State(0 .- 0*xinit.q, 1 .- 0*(xinit.p)))  # only optimize momenta
    mask_id = (mask .> 0.1) # get indices that correspond to momenta

    # initialisation
    Xᵒ = initSamplePath(tt_, xinit)
    Wᵒ = initSamplePath(tt_,  zeros(StateW, dwiener))
    Wnew = initSamplePath(tt_,  zeros(StateW, dwiener))
    if 1 in subsamples
        # push!(Xsave, copy(X))

    end
    x = deepvec(xinit)
    xᵒ = deepcopy(x)
    ∇x = deepcopy(x)
    ∇xᵒ = deepcopy(x)
    result = DiffResults.GradientResult(x) # allocate
    resultᵒ = DiffResults.GradientResult(xᵒ)


    if makefig
        xobs0comp1 = extractcomp(xobs0,1)
        xobs0comp2 = extractcomp(xobs0,2)
        xobsTcomp1 = extractcomp(xobsT,1)
        xobsTcomp2 = extractcomp(xobsT,2)
    end
    showmomenta = false

    # start iterations
    anim =    @animate for i in 2:ITER
        if makefig
            drawpath(i-1,x,X,objvals,x0,(xobs0comp1,xobs0comp2,xobsTcomp1,xobsTcomp2))
        end
        #   plotlandmarkpositions(X,P.n,model,xobs0,xobsT,nfs,db=2.6)

        # global ll, acc, X, Xᵒ, W, Wᵒ, Wnew, x, xᵒ, ∇x
        # global ∇xᵒ
        # global δ
        println("iteration $i")

        if sampler==:sgd
            δ = 0.01*ϵstep(i)
        end
        (x , W, X), ll, obj, acc  = updatepath!(X,Xᵒ,W,Wᵒ,Wnew,ll,x,xᵒ,∇x, ∇xᵒ,result, resultᵒ,
                                    sampler,Q,
                                    mask, mask_id, δ, ρ, acc)
        println()
        # save some of the results
        if i in subsamples
            #push!(Xsave, copy(X))
            push!(Xsave, convert_samplepath(X))
        end
        push!(objvals, obj)
        if makefig && (i==ITER)
            drawpath(ITER,x,X,objvals,x0,(xobs0comp1,xobs0comp2,xobsTcomp1,xobsTcomp2))
        end
    end

    fn = "me"*"_" * string(model) * "_" * string(sampler) *"_" * string(dataset)
    gif(anim, outdir*fn*".gif", fps = 20)
    mp4(anim, outdir*fn*".mp4", fps = 20)

    # drawobjective(objvals)

    perc_acc = 100acc/ITER
    println("Acceptance percentages (bridgepath - inital state): ",perc_acc)
    Xsave, objvals, perc_acc
end



function drawpath(i,x,X,objvals,x0,(xobs0comp1,xobs0comp2,xobsTcomp1,xobsTcomp2);showmomenta=false)
        s = deepvec2state(x).p
        s0 = x0.p # true momenta

        # plot initial and final shapes
        pp = Plots.plot(xobs0comp1, xobs0comp2,seriestype=:scatter, color=:black,label="q0", title="Landmark evolution")
        Plots.plot!(pp, repeat(xobs0comp1,2), repeat(xobs0comp2,2),seriestype=:path, color=:black,label="")
        Plots.plot!(pp, xobsTcomp1, xobsTcomp2,seriestype=:scatter , color=:orange,label="qT") # points move from black to orange
        Plots.plot!(pp, repeat(xobsTcomp1,2), repeat(xobsTcomp2,2),seriestype=:path, color=:orange,label="")

        if showmomenta
            Plots.plot!(pp, extractcomp(s,1), extractcomp(s,2),seriestype=:scatter ,
             color=:blue,label="p0 est") # points move from black to orange)
            Plots.plot!(pp, extractcomp(s0,1), extractcomp(s0,2),seriestype=:scatter ,
              color=:red,label="p0",markersize=5) # points move from black to orange)
              xlims!(-3,3)
              ylims!(-4,3)
        else
            xlims!(-3,3)
            ylims!(-4,2)
        end

        outg = [Any[X.tt[i], [X.yy[i][CartesianIndex(c, k)][l] for l in 1:d, c in 1:2]..., "point$k"] for k in 1:n, i in eachindex(X.tt) ][:]
        dfg = DataFrame(time=extractcomp(outg,1),pos1=extractcomp(outg,2),pos2=extractcomp(outg,3),mom1=extractcomp(outg,4),mom2=extractcomp(outg,5),pointID=extractcomp(outg,6))
        for j in 1:n
            #global pp
            el1 = dfg[:pointID].=="point"*"$j"
            dfg1 = dfg[el1,:]
            Plots.plot!(pp,dfg1[:pos1], dfg1[:pos2],label="")
        end

        pp2 = Plots.plot(collect(1:i), objvals[1:i],seriestype=:scatter ,color=:blue,markersize=1.5,label="",title="Loglikelihood approximation")
        Plots.plot!(pp2, collect(1:i), objvals[1:i] ,color=:blue,label="")
        xlabel!(pp2,"iteration")
        ylabel!(pp2,"stoch log likelihood")
        xlims!(0,ITER)

        l = @layout [a  b]
        Plots.plot(pp,pp2,background_color = :ivory,layout=l , size = (900, 500) )
end


function drawobjective(objvals)
    ITER = length(objvals)
    sc2 = Plots.plot(collect(1:ITER), objvals,seriestype=:scatter ,color=:blue,markersize=1.2,label="",title="Loglikelihood approximation")
    Plots.plot!(sc2, collect(1:ITER), objvals ,color=:blue,label="")
    xlabel!(sc2,"iteration")
    ylabel!(sc2,"stoch log likelihood")
    xlims!(0,ITER)
    display(sc2)
    png(sc2,"stochlogp.png")
end

"""
    Useful for storage of a samplepath of states
    Ordering is as follows:
    1) time
    2) landmark nr
    3) for each landmark: q1, q2 p1, p2

    With m time-steps, n landmarks, this entails a vector of length m * n * 2 * d
"""
function convert_samplepath(X)
    VA = VectorOfArray(map(x->deepvec(x),X.yy))
    vec(convert(Array,VA))
end


if TEST
    xinit = State(xobs0, zeros(PointF,n))
    xinit = State(xobs0, [Point(-1.0,3.0)/P.n for i in 1:P.n])
    ITER = 10
    lm_mcmc(tt_, (LT,ΣT,μT), (L0,Σ0), (xobs0,xobsT), P, Paux, model, sampler, dataset, xinit, δ, ITER; makefig=true)
end

# change parameter values and update
function updateguidedproposal!((α,γ), tt_, (Lt, Mt⁺ , μt, Ht), (LT,ΣT,μT), (L0, Σ0), (xobs0, xobsT), P::MarslandShardlow, Paux, Q)
    P = MarslandShardlow(α, γ, P.λ, P.n)
    if model == :ms
        Paux = MarslandShardlowAux(P, State(xobsT, mT))
    else
        Paux = LandmarksAux(P, State(xobsT, mT))
    end
    Q .= construct_guidedproposal!(tt_, (Lt, Mt⁺ , μt, Ht), (LT,ΣT,μT), (L0, Σ0), (xobs0, xobsT), P, Paux)
end
