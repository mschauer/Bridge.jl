StateW = PointF
dwiener = dimwiener(P)
if obs_atzero
    L0 = LT = [(i==j)*one(UncF) for i in 1:2:2P.n, j in 1:2P.n]
    Σ0 = ΣT = [(i==j)*σobs^2*one(UncF) for i in 1:P.n, j in 1:P.n]

else
    L0 = Array{UncF}(undef,0,2*P.n)
    Σ0 = Array{UncF}(undef,0,0)
    xobs0 = Array{PointF}(undef,0)
    LT = [(i==j)*one(UncF) for i in 1:2:2P.n, j in 1:2P.n]
    ΣT = [(i==j)*σobs^2*one(UncF) for i in 1:P.n, j in 1:P.n]
end
μT = zeros(PointF,P.n)


# now the new stuff:
nshapes = length(xobsTvec)
guidrecvec = [init_guidrec(tt_, LT, ΣT, μT, L0, Σ0, xobs0) for i in 1:nshapes]  # memory allocation for each shape
guidrecvecᵒ = [init_guidrec(tt_, LT, ΣT, μT, L0, Σ0, xobs0) for k in 1:nshapes]  # memory allocation for each shape
Pauxvec = [auxiliary(P,State(xobsTvec[k],mT)) for k in 1:nshapes] # auxiliary process for each shape
Qvec = [construct_guidedproposal!(tt_, guidrecvec[k], (LT,ΣT,μT), (L0, Σ0),
        (xobs0, xobsTvec[k]), P, Pauxvec[k]) for k in 1:nshapes]
Xvec = [initSamplePath(tt_, xinit) for i in 1:nshapes]
Wvec = [initSamplePath(tt_,  zeros(StateW, dwiener)) for i in 1:nshapes]
for i in 1:nshapes
    sample!(Wvec[i], Wiener{Vector{StateW}}())
end
ll = simguidedlm_llikelihood!(LeftRule(), Xvec, xinit, Wvec, Qvec; skip=sk)


Qvecᵒ = [construct_guidedproposal!(tt_, guidrecvec[i], (LT,ΣT,μT), (L0, Σ0),
        (xobs0, xobsTvec[i]), P, Pauxvec[i]) for i in 1:nshapes]

# saving objects
objvals = Float64[]  # keep track of (sgd approximation of the) loglikelihood
acc = zeros(4) # keep track of mcmc accept probs (first comp is for CN update; 2nd component for updates on initial momenta, 3rd parameter updates, 4th update on initial positions)
Xsave = typeof(zeros(length(tt_) * P.n * 2 * d * nshapes))[]
parsave = Vector{Float64}[]
#push!(Xsave, convert_samplepath(Xvec[1]))
push!(Xsave, convert_samplepath(Xvec))
push!(objvals, sum(ll))
push!(parsave,[P.a, P.c, getγ(P)])

# memory allocations
Xvecᵒ = [initSamplePath(tt_, xinit)  for i in 1:nshapes]
Wᵒ = initSamplePath(tt_,  zeros(StateW, dwiener))
Wnew = initSamplePath(tt_,  zeros(StateW, dwiener))
x = deepvec(xinit)
xᵒ = deepcopy(x)
∇x = deepcopy(x)
∇xᵒ = deepcopy(x)
llout = copy(ll)
lloutᵒ = copy(ll)


ptemp = zeros(PointF,P.n)


for i in 1:ITER
    global acc
    global P
    # updates paths
    update_path!(Xvec, Xvecᵒ, Wvec, Wᵒ, Wnew, ll, x, Qvec, ρ, acc)


    # update initial state
    #    updatekernel can be :mala_pos, :mala_mom, :mala_posandmom, :lmforward_pos
    #updatekernel = sample([:mala_mom,:lmforward_pos])
    obj = update_initialstate!(Xvec,Xvecᵒ,Wvec,ll,x,xᵒ,∇x, ∇xᵒ,llout, lloutᵒ,
                        sampler, Qvec, δ, acc, :lmforward_pos, ptemp)

    obj = update_initialstate!(Xvec,Xvecᵒ,Wvec,ll,x,xᵒ,∇x, ∇xᵒ,llout, lloutᵒ,
                                             sampler, Qvec, δ, acc, :mala_mom, ptemp)

    # update parameters
    P, acc= update_pars(P, tt_, mT, guidrecvecᵒ, (LT,ΣT,μT), (L0, Σ0),
                Xvec, Xvecᵒ,Wvec, Qvec, Qvecᵒ, x, ll, (prior_a, prior_c, prior_γ), (σ_a,σ_c,σ_γ), acc)

    println()
    # save some of the results
    if i in subsamples
        #push!(Xsave, convert_samplepath(Xvec[1]))
        push!(Xsave, convert_samplepath(Xvec))
        push!(parsave, [P.a, P.c, getγ(P)])
        push!(objvals, obj)
    end
end
