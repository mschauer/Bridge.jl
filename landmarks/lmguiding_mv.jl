# first run lm_main.jl
import Bridge: kernelr3!, R3!, target, auxiliary, constdiff, llikelihood, _b!, B!, σ!, b!

"""
    GuidRecursions defines a struct that contains all info required for computing the guiding term and
    likelihood (including ptilde term) for a single shape
"""
mutable struct GuidRecursions{TL,TM⁺,TM, Tμ, TH, TLt0, TMt⁺0, Tμt0}
    Lt::Vector{TL} # Lt on grid tt
    Mt⁺::Vector{TM⁺}  # Mt⁺ on grid tt
    Mt::Vector{TM}  # Mt on grid tt
    μt::Vector{Tμ}  # μt on grid tt
    Ht::Vector{TH}  # Ht on grid tt
    Lt0::TLt0      # Lt at time 0, after gpupdate step incorporating observation xobs0
    Mt⁺0::TMt⁺0    # inv(Mt) at time 0, after gpupdate step incorporating observation xobs0
    μt0::Tμt0       # μt at time 0, after gpupdate step incorporating observation xobs0

    function GuidRecursions(Lt, Mt⁺,Mt, μt, Ht, Lt0, Mt⁺0, μt0)
            new{eltype(Lt), eltype(Mt⁺), eltype(Mt),eltype(μt),eltype(Ht), typeof(Lt0), typeof(Mt⁺0), typeof(μt0)}(Lt, Mt⁺,Mt, μt, Ht, Lt0, Mt⁺0, μt0)
    end
end

"""
Construct guided proposal on a single segment for a single shape
"""
struct GuidedProposal!{T,Ttarget,Taux,TL,Txobs0,TxobsT,F} <: ContinuousTimeProcess{T}
    target::Ttarget   # P
    aux::Taux      # Ptilde
    tt::Vector{Float64}  # grid of time points on single segment (S,T]
    guidrec::TL # guided recursions on grid tt
    xobs0::Txobs0   # observation at time 0
    xobsT::TxobsT  # observation at time T
    endpoint::F

    function GuidedProposal!(target, aux, tt_, guidrec, xobs0, xobsT,  endpoint=Bridge.endpoint)
        tt = collect(tt_)
        new{Bridge.valtype(target),typeof(target),typeof(aux),typeof(guidrec),typeof(xobs0),typeof(xobsT),typeof(endpoint)}(target, aux, tt, guidrec, xobs0, xobsT, endpoint)
    end
end

"""
    Initialise (allocate memory) a struct of type GuidRecursions for a single shape

    guidres = init_guidrec(tt_, LT, ΣT, μT, L0, Σ0, xobs0)
"""
function init_guidrec(t,LT,ΣT::Array{Pnt,2},μT,L0,Σ0,xobs0) where Pnt
    Lt =  [copy(LT) for s in t]
    Mt⁺ = [copy(ΣT) for s in t]
    Mt = map(X->InverseCholesky(lchol(X)),Mt⁺)
    μt = [copy(μT) for s in t]
    H = LT' * (ΣT * LT )
    Ht = [copy(H) for s in t]
    Lt0 = copy([L0; LT])

    m = size(Σ0)[1]
    n = size(ΣT)[2]
    if m==0
        Mt⁺0 = copy(ΣT)
    else
        Mt⁺0 = [copy(Σ0) zeros(Pnt,m,n); zeros(Pnt,n,m) copy(ΣT)]
    end
    μt0 = [0*xobs0; copy(μT)]
    GuidRecursions(Lt, Mt⁺, Mt, μt, Ht, Lt0, Mt⁺0, μt0)
end

"""
    Guided proposal update for newly incoming observation at time zero.
    Information on new observations at time zero is (L0, Σ0, xobs0)
    Values just after time zero, (Lt0₊, Mt⁺0₊, μt0₊) are updated to time zero, the result being
    written into (Lt0, Mt⁺0, μt0)
"""
function lm_gpupdate!(Lt0₊, Mt⁺0₊::Array{Pnt,2}, μt0₊, (L0, Σ0, xobs0), Lt0, Mt⁺0, μt0) where Pnt
    Lt0 .= [L0; Lt0₊]
    m = size(Σ0)[1]
    n = size(Mt⁺0₊)[2]
    if m==0
        Mt⁺0 .= Mt⁺0₊
    else
        Mt⁺0 .= [Σ0 zeros(Pnt,m,n); zeros(Pnt,n,m) Mt⁺0₊]
    end
    μt0 .= [0*xobs0; μt0₊]
end

"""
    Construct guided proposal for a single shape on grid tt_
"""
function construct_guidedproposal!(tt_, guidrec, (LT,ΣT,μT), (L0, Σ0), (xobs0, xobsT), P, Paux)
    Lt0₊, Mt⁺0₊, μt0₊ =  guidingbackwards!(Lm(), tt_, (guidrec.Lt, guidrec.Mt⁺,guidrec.μt), Paux, (LT, ΣT, μT))
    lm_gpupdate!(Lt0₊, Mt⁺0₊, μt0₊, (L0, Σ0, xobs0),guidrec.Lt0, guidrec.Mt⁺0, guidrec.μt0)
    guidrec.Mt = map(X -> InverseCholesky(lchol(X)),guidrec.Mt⁺)
    for i in 1:length(tt_)
        guidrec.Ht[i] .= guidrec.Lt[i]' * (guidrec.Mt[i] * guidrec.Lt[i] )
    end
    GuidedProposal!(P, Paux, tt_, guidrec, xobs0, xobsT)
end

##########################
struct Lm  end

"""
    Solve backwards recursions in L, M, μ parametrisation on grid t
    implicit: if true, Euler backwards is used for solving ODE for Lt, else Euler forwards is used

    Case lowrank=true still gives an error: fixme!
"""
function guidingbackwards!(::Lm, t, (Lt, Mt⁺, μt), Paux, (LT, ΣT , μT); implicit=true, lowrank=false)
    Mt⁺[end] .= ΣT
    Lt[end] .= LT
    μt[end] .= μT

    BB = Bridge.B(0, Paux)          # does not depend on time
    β = vec(Bridge.β(0,Paux))       # does not depend on time

    # various ways to compute ã (which does not depend on time);
    aa = Bridge.a(0, Paux) # vanilla, no (possibly enclose with Matrix)
    if !lowrank
        oldtemp = 0.5* Lt[end]* aa * Matrix(Lt[end]') * dt
    else
        #aalr = pheigfact(deepmat(aa), rtol=1e-8)  # control accuracy of lr approx
        aalr = pheigfact(deepmat(aa))  # control accuracy of lr approx
        # println("Rank ",size(aalr[:vectors],2), " approximation to ã")
        sqrt_aalr = deepmat2unc(aalr[:vectors] * diagm(0=> sqrt.(aalr[:values])))
    end

    for i in length(t)-1:-1:1
        dt = t[i+1]-t[i]
        if implicit
            Lt[i] .= Lt[i+1]/(I - dt* BB)
        else
            Lt[i] .=  Lt[i+1] * (I + BB * dt)
        end
        if !lowrank
            temp = 0.5 * Lt[i]* aa * Matrix(Lt[i]') * dt
            Mt⁺[i] .= Mt⁺[i+1] + oldtemp + temp
            oldtemp = temp
        else
            C = (0.5 * dt) * Bridge.outer(Lt[i+1] * sqrt_aalr)
            Mt⁺[i] .= Mt⁺[i+1] + (C + C')
        end
        μt[i] .= μt[i+1] + 0.5 * (Lt[i] + Lt[i+1]) * β * dt  # trapezoid rule
    end
    (Lt[1], Mt⁺[1], μt[1])
end

target(Q::GuidedProposal!) = Q.target
auxiliary(Q::GuidedProposal!) = Q.aux
constdiff(Q::GuidedProposal!) = constdiff(target(Q)) && constdiff(auxiliary(Q))

"""
    Evaluate drift bᵒ of guided proposal at (t,x), write into out
"""
function _b!((i,t), x::State, out::State, Q::GuidedProposal!)
    Bridge.b!(t, x, out, Q.target)
    out .+= amul(t,x,Q.guidrec.Lt[i]' * (Q.guidrec.Mt[i] *(Q.xobsT-Q.guidrec.μt[i]-Q.guidrec.Lt[i]*vec(x))),Q.target)
    out
end

"""
    Evaluate σ(t,x) dW and write into out
"""
σ!(t, x, dw, out, Q::GuidedProposal!) = σ!(t, x, dw, out, Q.target)

function _r!((i,t), x::State, out::State, Q::GuidedProposal!)
    out .= vecofpoints2state(Q.guidrec.Lt[i]' * (Q.guidrec.Mt[i] *(Q.xobsT-Q.guidrec.μt[i]-Q.guidrec.Lt[i]*vec(x))))
    out
end


function guidingterm((i,t),x,Q::GuidedProposal!)
    #Bridge.b(t,x,Q.target) +
    amul(t,x,Q.guidrec.Lt[i]' * (Q.guidrec.Mt[i] *(Q.xobsT-Q.guidrec.μt[i]-Q.guidrec.Lt[i]*vec(x))),Q.target)
end
"""
Returns the guiding terms a(t,x)*r̃(t,x) along the path of a guided proposal
for each value in X.tt.
Hence, it returns an Array of type State
"""
function guidingterms(X,Q::GuidedProposal!)
    i = first(1:length(X.tt))
    out = [guidingterm((i,X.tt[i]),X.yy[i],Q)]
    for i in 2:length(X.tt)
        push!(out, guidingterm((i,X.tt[i]),X.yy[i],Q))
    end
    out
end

"""
    Compute log ρ(0,x_0)
"""
function lρtilde(x0, Q)
  y = deepvec([Q.xobs0; Q.xobsT] - Q.guidrec.μt0 - Q.guidrec.Lt0*vec(x0))
  M⁺0deep = deepmat(Q.guidrec.Mt⁺0)
  -0.5*logdet(M⁺0deep) -0.5*dot(y, M⁺0deep\y)
end

"""
    Simulate guided proposal and compute loglikelihood for one shape

    solve sde inplace and return loglikelihood (thereby avoiding 'double' computations)
"""
function simguidedlm_llikelihood!(::LeftRule,  Xᵒ, x0, W, Q::GuidedProposal!; skip = 0, ll0 = true)
    Pnt = eltype(x0)
    tt =  Xᵒ.tt
    Xᵒ.yy[1] .= deepvalue(x0)
    som::deepeltype(x0)  = 0.

    # initialise objects to write into
    # srout and strout are vectors of Points
    dwiener = dimwiener(Q.target)
    srout = zeros(Pnt, dwiener)
    strout = zeros(Pnt, dwiener)

    x = copy(x0)

    rout = copy(x0)
    bout = copy(x0)
    btout = copy(x0)
    wout = copy(x0)

    if !constdiff(Q)
        At = Bridge.a((1,0), x0, auxiliary(Q))  # auxtimehomogeneous switch
        A = zeros(Unc{deepeltype(x0)}, 2Q.target.n,2Q.target.n)
    end

    for i in 1:length(tt)-1
        dt = tt[i+1]-tt[i]
        b!(tt[i], x, bout, target(Q)) # b(t,x)
        _r!((i,tt[i]), x, rout, Q) # tilder(t,x)
        σt!(tt[i], x, rout, srout, target(Q))      #  σ(t,x)' * tilder(t,x) for target(Q)
        Bridge.σ!(tt[i], x, srout*dt + W.yy[i+1] - W.yy[i], wout, target(Q)) # σ(t,x) (σ(t,x)' * tilder(t,x) + dW(t))


        # likelihood terms
        if i<=length(tt)-1-skip
            _b!((i,tt[i]), x, btout, auxiliary(Q))
            som += dot(bout-btout, rout) * dt
            if !constdiff(Q)
                σt!(tt[i], x, rout, strout, auxiliary(Q))  #  tildeσ(t,x)' * tilder(t,x) for auxiliary(Q)
                som += 0.5*Bridge.inner(srout) * dt    # |σ(t,x)' * tilder(t,x)|^2
                som -= 0.5*Bridge.inner(strout) * dt   # |tildeσ(t,x)' * tilder(t,x)|^2
                Bridge.a!((i,tt[i]), x, A, target(Q))
                som += 0.5*(dot(At,Q.guidrec.Ht[i]) - dot(A,Q.guidrec.Ht[i])) * dt
            end
        end
        x .= x + dt * bout + wout
        Xᵒ.yy[i+1] .= deepvalue(x)
    end
    if ll0
        logρ0 = lρtilde(x0,Q)
    else
        logρ0 = 0.0 # don't compute
    end
    copyto!(Xᵒ.yy[end], Bridge.endpoint(Xᵒ.yy[end],Q))
    som + logρ0
end

"""
    Simulate guided proposal and compute loglikelihood (vector version, multiple shapes)

    solve sde inplace and return loglikelihood (thereby avoiding 'double' computations)

    simguidedlm_llikelihood!(LeftRule(), Xvec, xinit, Wvec, Qvec; skip=sk)
"""
function simguidedlm_llikelihood!(::LeftRule,  Xvec, x0, Wvec, Qvec::Vector; skip = 0, ll0 = true) # rather would like to dispatch on type and remove '_mv' from function name
    nshapes = length(Qvec)
    soms  = zeros(deepeltype(x0), nshapes)
    for k in 1:nshapes
        soms[k] = simguidedlm_llikelihood!(LeftRule(), Xvec[k],x0,Wvec[k],Qvec[k] ;skip=skip,ll0=ll0)
    end
    soms
end


# convert dual to float, while retaining float if type is float
deepvalue(x::Float64) = x
deepvalue(x::ForwardDiff.Dual) = ForwardDiff.value(x)
deepvalue(x) = deepvalue.(x)
function deepvalue(x::State)
    State(deepvalue.(x.x))
end

"""
    update bridges using Crank-Nicholsen scheme with parameter ρ (only in case the method is mcmc)
    Newly accepted bridges are written into (X,W), loglikelihood on each segment is written into vector ll

    update_path!(Xvec,Xvecᵒ,Wvec,Wᵒ,Wnew,ll,x, Qvec, ρ, acc)
"""
function update_path!(Xvec,Xᵒ,Wvec,Wᵒ,Wnew,ll,x, Qvec, ρ, acc)
    nshapes = length(Xvec)
    nn = length(Xvec[1].yy)
    x0 = deepvec2state(x)

    # Uvec = deepcopy(Xvec)
    # llU = simguidedlm_llikelihood!(LeftRule(), Uvec, x0, Wvec, Qvec)
    # if norm(ll - llU) > 0.01 || norm(Xvec[1].yy-Uvec[1].yy)> 0.01
    #     println("inconsistency in input of update_path registered.")
    #     println(norm(Xvec[1].yy-Uvec[1].yy))
    #     @info "Done."*"\x7"^6
    # end


    # From current state (x,W) with loglikelihood ll, update to (x, Wᵒ)
    for k in 1:nshapes
        sample!(Wnew, Wiener{Vector{PointF}}())
        for i in 1:nn
            Wᵒ.yy[i] .= ρ * Wvec[k].yy[i] + sqrt(1-ρ^2) * Wnew.yy[i]
        end
        llᵒ_ = simguidedlm_llikelihood!(LeftRule(), Xᵒ, x0, Wᵒ, Qvec[k];skip=sk)

#        println("norm(Xᵒ.yy-Xvec[k].yy) ", norm(Xᵒ.yy-Xvec[k].yy))
        diff_ll = llᵒ_ - ll[k]
        if log(rand()) <= diff_ll
            for i in 1:nn
                Xvec[k].yy[i] .= Xᵒ.yy[i]
                Wvec[k].yy[i] .= Wᵒ.yy[i]
            end
            println("update innovation. diff_ll: ",round(diff_ll;digits=6),"  accepted")
            ll[k] = llᵒ_
            acc[1] +=1
        else
            println("update innovation. diff_ll: ",round(diff_ll;digits=6),"  rejected")
        end
    end
    Xvec, Wvec, ll
    #nothing
end


"""
    Stochastic approximation for loglikelihood.

    Simulate guided proposal and compute loglikelihood for starting point x0,
    guided proposals defined by Qvec and Wiener increments in Wvec.
    Guided proposals are written into Xvec.
    Writes vector of loglikelihoods into llout.
    Returns sum of loglikelihoods
"""
function slogρ!(x0deepv, Qvec, Wvec,Xvec,llout) # stochastic approx to log(ρ)
    x0 = deepvec2state(x0deepv)
    lltemp = simguidedlm_llikelihood!(LeftRule(), Xvec, x0, Wvec, Qvec; skip=sk)#overwrites Xvec
    llout .= ForwardDiff.value.(lltemp)
    sum(lltemp)
end
slogρ!(Q, W, X, llout) = (x) -> slogρ!(x, Q, W,X,llout)



function update_pars!(guidrecvecᵒ, (LT,ΣT,μT), (L0, Σ0),
            Xvec, Xvecᵒ,Wvec, Qvec, Qvecᵒ, x, ll, (prior_a, prior_c, prior_γ), (σ_a,σ_c,σ_γ), acc)
    #println("startf ",ll)
    nshapes = length(Xvec)
    P = Qvec[1].target
    tt_ = Xvec[1].tt

    aᵒ = P.a * exp(σ_a * randn())
    cᵒ = P.c * exp(σ_c * randn())
    γᵒ = getγ(P) * exp(σ_γ * randn())
    if isa(P,MarslandShardlow)
        Pᵒ = MarslandShardlow(aᵒ,cᵒ,γᵒ,P.λ, P.n)
    elseif isa(P,Landmarks)
        nfs = construct_nfs(P.db, P.nfstd, γᵒ) # need ot add db and nfstd to struct Landmarks
        Pᵒ = Landmarks(aᵒ,cᵒ,P.n,P.db,P.nfstd,nfs)
    end
    Pauxvecᵒ = [auxiliary(Pᵒ,Qvec[k].aux.xT) for k in 1:nshapes] # auxiliary process for each shape
    for k in 1:nshapes
        Qvecᵒ[k] = construct_guidedproposal!(tt_, guidrecvecᵒ[k], (LT,ΣT,μT), (L0, Σ0),
                                (Qvec[k].xobs0, Qvec[k].xobsT), Pᵒ, Pauxvecᵒ[k])
    end
    llᵒ = simguidedlm_llikelihood!(LeftRule(), Xvecᵒ, deepvec2state(x), Wvec, Qvecᵒ; skip=sk)
    A = logpdf(prior_a,aᵒ) - logpdf(prior_a,P.a) +
        logpdf(prior_c,cᵒ) - logpdf(prior_c,P.c) +
        logpdf(prior_γ,γᵒ) - logpdf(prior_γ,getγ(P)) +
        sum(llᵒ) - sum(ll) +
        logpdf(LogNormal(log(Pᵒ.a),σ_a),P.a)- logpdf(LogNormal(log(P.a),σ_a),Pᵒ.a)+
        logpdf(LogNormal(log(Pᵒ.c),σ_c),P.c)- logpdf(LogNormal(log(P.c),σ_c),Pᵒ.c)+
        logpdf(LogNormal(log(getγ(Pᵒ)),σ_γ),getγ(P))- logpdf(LogNormal(log(getγ(P)),σ_γ),getγ(Pᵒ))

    if log(rand()) <= A  # assume symmetric proposal and uniform prior, adjust later
        println("logaccept for parameter update ", round(A;digits=4), "  accepted")
        acc[3] +=1
        #println("endf ",llᵒ, norm(Xvecᵒ[2].yy))
        println()
        #return( (Xvecᵒ,Qvecᵒ,Wvec,Pᵒ,llᵒ,acc, simguidedlm_llikelihood!(LeftRule(), Xvecᵒ, deepvec2state(x), Wvec, Qvecᵒ; skip=sk)))
        # P = Pᵒ
        for k in 1:nshapes
            for i in 1:length(Xvec[1].yy)
                Xvec[k].yy[i] .= Xvecᵒ[k].yy[i]
            end
        end
        boolacc = true
    else
        println("logaccept for parameter update ", round(A;digits=4), "  rejected")
        #println("endf ",ll, norm(Xvec[2].yy))
        boolacc = false
        #return( (Xvec,Qvec,Wvec,P,ll,acc,simguidedlm_llikelihood!(LeftRule(), Xvec, deepvec2state(x), Wvec, Qvec; skip=sk)))
    end
    boolacc, Qvecᵒ, llᵒ
end



"""
    Perform mcmc or sgd for landmarks model using the LM-parametrisation
    tt_:      time grid
    (xobs0,xobsT): observations at times 0 and T
    σobs: standard deviation of Gaussian noise assumed on xobs0 and xobsT
    mT: vector of momenta at time T used for constructing guiding term
    P: target process

    sampler: either sgd (stochastic gradient descent) or mcmc (Markov Chain Monte Carlo)
    dataset: dataset to extract xobs0 and xobsT
    xinit: initial guess on starting state

    ITER: number of iterations
    subsamples: vector of indices of iterations that are to be saved

    δ: parameter for Langevin updates on initial state
    prior_a: prior on parameter a
    prior_c: prior on parameter c
    prior_γ: prior on parameter γ
    σ_a: parameter determining update proposal for a [update a to aᵒ as aᵒ = a * exp(σ_a * rnorm())]
    σ_c: parameter determining update proposal for c [update c to cᵒ as cᵒ = c * exp(σ_c * rnorm())]
    σ_γ: parameter determining update proposal for γ [update γ to γᵒ as γᵒ = γ * exp(σ_γ * rnorm())]

    outdir: output directory for animation
    pb:: Lmplotbounds (axis used for plotting landmarks evolution)
    updatepars: logical flag for updating pars a, c, γ
    makefig: logical flag for making figures
    showmomenta: logical flag if momenta are also drawn in figures

    Returns:
    Xsave: saved iterations of all states at all times in tt_
    parsave: saved iterations of all parameter updates ,
    objvals: saved values of stochastic approximation to loglikelihood
    perc_acc: acceptance percentages (bridgepath - inital state)


    anim, Xsave, parsave, objvals, perc_acc = lm_mcmc_mv(tt_, (xobs0,xobsTvec), σobs, mT, P,
             sampler, dataset, obs_atzero,
             xinit, ITER, subsamples,
            (δ, prior_a, prior_c, prior_γ, σ_a, σ_c, σ_γ),
            outdir, pb; updatepars = true, makefig=true, showmomenta=false)
"""
function lm_mcmc(tt_, (xobs0,xobsTvec), σobs, mT, P,
         sampler, dataset, obs_atzero,
         xinit, ITER, subsamples,
        (ρ, δ, prior_a, prior_c, prior_γ, σ_a, σ_c, σ_γ),
        outdir, pb; updatepars = true, makefig=true, showmomenta=false)

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
    Qvecᵒ = deepcopy(Qvec)
    Xvec = [initSamplePath(tt_, xinit) for i in 1:nshapes]
    Wvec = [initSamplePath(tt_,  zeros(StateW, dwiener)) for i in 1:nshapes]
    for i in 1:nshapes
        sample!(Wvec[i], Wiener{Vector{StateW}}())
    end
    ll = simguidedlm_llikelihood!(LeftRule(), Xvec, xinit, Wvec, Qvec; skip=sk)

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
    Xvecᵒ = [initSamplePath(tt_, xinit)  for k in 1:nshapes]
    Qvecᵒ = deepcopy(Qvec)
    Xᵒ = initSamplePath(tt_, xinit)

    Wᵒ = initSamplePath(tt_,  zeros(StateW, dwiener))
    Wnew = initSamplePath(tt_,  zeros(StateW, dwiener))
    x = deepvec(xinit)
    xᵒ = deepcopy(x)
    ∇x = deepcopy(x)
    ∇xᵒ = deepcopy(x)
    llout = deepcopy(ll)
    lloutᵒ = deepcopy(ll)

    if makefig
        #plotlandmarkpositions(X,P,xobs0,xobsT;db=4)
        xobs0comp1 = extractcomp(xobs0,1)
        xobs0comp2 = extractcomp(xobs0,2)
        xobsTcomp1 = extractcomp(xobsTvec[1],1)
        xobsTcomp2 = extractcomp(xobsTvec[1],2)
        pp1 = plotshapes(xobs0comp1,xobs0comp2,xobsTcomp1, xobsTcomp2)
    end

    ptemp = zeros(PointF,P.n)
    initstate_accinfo = []

#    for i in 1:ITER
    anim =    @animate for i in 1:ITER
        if makefig
            drawpath(i-1,P.n,x,Xvec[1],objvals,parsave,(xobs0comp1,xobs0comp2,xobsTcomp1, xobsTcomp2),pb)
        end
        println()
        println("iteration $i")

        # updates paths
        Xvec, Wvec, ll = update_path!(Xvec, Xᵒ, Wvec, Wᵒ, Wnew, ll, x, Qvec, ρ, acc)

        # update initial state
        boolacc, obj, accinfo, xacc, ∇xacc, llout = update_initialstate!(Xvec,Xvecᵒ,Wvec,ll,x,xᵒ,∇x, ∇xᵒ,llout, lloutᵒ,
                                                      sampler, Qvec, δ, acc, :mala_mom, ptemp, i)
        push!(initstate_accinfo, accinfo)
        ll .= llout
        if boolacc
            x .= xacc
            ∇x .= ∇xacc
        end

        boolacc, obj, accinfo, xacc, ∇xacc, llacc = update_initialstate!(Xvec,Xvecᵒ,Wvec,ll,x,xᵒ,∇x, ∇xᵒ,llout, lloutᵒ,
                                                      sampler, Qvec, δ, acc, :lmforward, ptemp, i)
        push!(initstate_accinfo, accinfo)
        ll .= llacc
        if boolacc
            x .= xacc
            ∇x .= ∇xacc
        end



        # update parameters
        boolacc, Qvecacc, llout = update_pars!(guidrecvecᵒ, (LT,ΣT,μT), (L0, Σ0),
                    Xvec, Xvecᵒ,Wvec, Qvec, Qvecᵒ, x, ll, (prior_a, prior_c, prior_γ), (σ_a,σ_c,σ_γ), acc)
        if boolacc
            ll .= llout
            Qvec = deepcopy(Qvecacc) # seems crucial to have a deepcopy here
        end

        # save some of the results
        P = Qvec[1].target
        if i in subsamples
            push!(Xsave, convert_samplepath(Xvec))
            push!(parsave, [P.a, P.c, getγ(P)])
            push!(objvals, obj)
        end
        #
        if makefig && (i==ITER)
            drawpath(ITER,P.n,x,Xvec[1],objvals,parsave,(xobs0comp1,xobs0comp2,xobsTcomp1, xobsTcomp2),pb)
        end
    end
    perc_acc = 100acc/(nshapes*ITER)
    println("Acceptance percentages (bridgepath - inital state momenta - parameters - initial state positions): ",perc_acc)
    anim, Xsave, parsave, objvals, perc_acc, initstate_accinfo
end
