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
    # should check what happens when there is no observation at time zero!!!
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



######################
##########################
struct Lm  end

"""
    Solve backwards recursions in L, M, μ parametrisation on grid t
    implicit: if true, Euler backwards is used for solving ODE for Lt, else Euler forwards is used
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
            Mt⁺[i] .= Mt⁺[i+1] + (C+C')
        end
        μt[i] .= μt[i+1] + 0.5 * (Lt[i] + Lt[i+1]) * β * dt  # trapezoid rule
    end
    (Lt[1], Mt⁺[1], μt[1])
end

target(Q::GuidedProposal!) = Q.target
auxiliary(Q::GuidedProposal!) = Q.aux
constdiff(Q::GuidedProposal!) = constdiff(target(Q)) && constdiff(auxiliary(Q))

function _b!((i,t), x::State, out::State, Q::GuidedProposal!)
    Bridge.b!(t, x, out, Q.target)
    out .+= amul(t,x,Q.guidrec.Lt[i]' * (Q.guidrec.Mt[i] *(Q.xobsT-Q.guidrec.μt[i]-Q.guidrec.Lt[i]*vec(x))),Q.target)
    out
end

σ!(t, x, dw, out, Q::GuidedProposal!) = σ!(t, x, dw, out, Q.target)

function _r!((i,t), x::State, out::State, Q::GuidedProposal!)
    out .= vecofpoints2state(Q.guidrec.Lt[i]' * (Q.guidrec.Mt[i] *(Q.xobsT-Q.guidrec.μt[i]-Q.guidrec.Lt[i]*vec(x))))
    out
end
# need function that multiplies square unc with state and outputs state

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

function lptilde_mv(x, Q)
  y = deepvec([Q.xobs0; Q.xobsT] - Q.guidrec.μt0 - Q.guidrec.Lt0*vec(x))
  M⁺0deep = deepmat(Q.guidrec.Mt⁺0)
  -0.5*logdet(M⁺0deep) -0.5*dot(y, M⁺0deep\y)
end



"""
    Simulate guided proposal and compute loglikelihood

    solve sde inplace and return loglikelihood (thereby avoiding 'double' computations)
"""
function simguidedlm_llikelihood!(::LeftRule,  Xᵒ, x0, W, Q::GuidedProposal!; skip = 0, ll0 = true)
    Pnt = eltype(x0)
    tt =  Xᵒ.tt
    Xᵒ.yy[1] .= deepvalue(x0)
    x = copy(x0)
    som::deepeltype(x0)  = 0.

    # initialise objects to write into
    # srout and strout are vectors of Points
    dwiener = dimwiener(Q.target)
    srout = zeros(Pnt, dwiener)
    strout = zeros(Pnt, dwiener)

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
        x.= x + dt * bout +  wout
        Xᵒ.yy[i+1] .= deepvalue(x)
    end
    if ll0
        logρ0 = lptilde_mv(x0,Q)
    else
        logρ0 = 0.0 # don't compute
    end
    copyto!(Xᵒ.yy[end], Bridge.endpoint(Xᵒ.yy[end],Q))
    som + logρ0
end

"""
    Simulate guided proposal and compute loglikelihood (vector version)

    solve sde inplace and return loglikelihood (thereby avoiding 'double' computations)

    simguidedlm_llikelihood!(LeftRule(), Xvec, xinit, Wvec, Qvec; skip=sk)
"""
function simguidedlm_llikelihood!(::LeftRule,  Xvecᵒ, x0, Wvec, Qvec::Vector; skip = 0, ll0 = true) # rather would like to dispatch on type and remove '_mv' from function name
    nshapes = length(Xvecᵒ)
    Pnt = eltype(x0)
    tt =  Xvecᵒ[1].tt
    for k in 1:nshapes
        Xvecᵒ[k].yy[1] .= deepvalue(x0)
    end
    x = copy(x0)
    som::deepeltype(x0)  = 0.
    somvec = [copy(som) for i in 1:nshapes]

    # initialise objects to write into
    # srout and strout are vectors of Points
    dwiener = dimwiener(Qvec[1].target)
    srout = zeros(Pnt, dwiener)
    strout = zeros(Pnt, dwiener)

    rout = copy(x0)
    bout = copy(x0)
    btout = copy(x0)
    wout = copy(x0)

    for k in 1:nshapes
        Q = Qvec[k]
        if !constdiff(Q)
            At = Bridge.a((1,0), x0, auxiliary(Q))  # auxtimehomogeneous switch
            A = zeros(Unc{deepeltype(x0)}, 2Q.target.n,2Q.target.n)
        end

        for i in 1:length(tt)-1
            dt = tt[i+1]-tt[i]
            b!(tt[i], x, bout, target(Q)) # b(t,x)
            _r!((i,tt[i]), x, rout, Q) # tilder(t,x)
            σt!(tt[i], x, rout, srout, target(Q))      #  σ(t,x)' * tilder(t,x) for target(Q)
            Bridge.σ!(tt[i], x, srout*dt + Wvec[k].yy[i+1] - Wvec[k].yy[i], wout, target(Q)) # σ(t,x) (σ(t,x)' * tilder(t,x) + dW(t))

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
            x.= x + dt * bout +  wout
            Xvecᵒ[k].yy[i+1] .= deepvalue(x)
        end
        if ll0
            logρ0 = lptilde_mv(x0,Q)
        else
            logρ0 = 0.0 # don't compute
        end
        copyto!(Xvecᵒ[k].yy[end], Bridge.endpoint(Xvecᵒ[k].yy[end],Q))
        somvec[k] = som + logρ0
        som .* 0
    end
    somvec
end


### below: function from automaticdiff_lm split up, making that file obsolete

# convert dual to float, while retaining float if type is float
deepvalue(x::Float64) = x
deepvalue(x::ForwardDiff.Dual) = ForwardDiff.value(x)
deepvalue(x) = deepvalue.(x)
function deepvalue(x::State)
    State(deepvalue.(x.x))
end



"""
    update bridges (only in case the method is mcmc)

    W, X, ll, acc = update_path!(X,Xᵒ,W,Wᵒ,Wnew,ll,x,sampler, Qvec,mask, mask_id, δ, ρ, acc)
"""
function update_path!(Xvec,Xvecᵒ,Wvec,Wᵒ,Wnew,ll,x,sampler, Qvec, ρ, acc)
    nshapes = length(Xvec)
    if sampler==:mcmc
        # From current state (x,W) with loglikelihood ll, update to (x, Wᵒ)
        for k in 1:nshapes
            sample!(Wnew, Wiener{Vector{PointF}}())
            Wᵒ.yy .= ρ * Wvec[k].yy + sqrt(1-ρ^2) * Wnew.yy # can get rid of Wnew (inplace)
            llᵒ = simguidedlm_llikelihood!(LeftRule(), Xvecᵒ[k], deepvec2state(x), Wᵒ, Qvec[k];skip=sk)
            if log(rand()) <= llᵒ - ll[k]
                for i in eachindex(Xvec[k].yy)
                    Xvec[k].yy[i] .= Xvecᵒ[k].yy[i]
                    Wvec[k].yy[i] .= Wᵒ.yy[i]
                end
                ll[k] = llᵒ
                println("update innovation: ll $ll[k] $llᵒ, diff_ll: ",round(llᵒ-ll[k];digits=3),"  accepted")
                #boolacc = true
                acc[1] +=1
            else
                println("update innovation: ll[k] $ll $llᵒ, diff_ll: ",round(llᵒ-ll[k];digits=3),"  rejected")
            end
        end
    end
    nothing
end

# ORIGINAL VERSION
function slogρ_mv(x0deepv, Qvec, Wvec,Xvec) # stochastic approx to log(ρ)
    x0 = deepvec2state(x0deepv)
    lltemp = simguidedlm_llikelihood!(LeftRule(), Xvec, x0, Wvec, Qvec; skip=sk)#overwrites Xvec

    sum(lltemp)
end
slogρ_mv(Q, W, X) = (x) -> slogρ_mv(x, Q, W,X)


function slogρ_mv!(x0deepv, Qvec, Wvec,Xvec,llout) # stochastic approx to log(ρ)
    x0 = deepvec2state(x0deepv)
    lltemp = simguidedlm_llikelihood!(LeftRule(), Xvec, x0, Wvec, Qvec; skip=sk)#overwrites Xvec
    llout .= ForwardDiff.value.(lltemp)
    sum(lltemp)
end
slogρ_mv!(Q, W, X, llout) = (x) -> slogρ_mv!(x, Q, W,X,llout)


"""
    update initial state

    x , W, X, ll, obj, acc = update_initialstate!(X,Xᵒ,W,ll,x,xᵒ,∇x, ∇xᵒ,result, resultᵒ,
                    sampler, Q,mask, mask_id, δ, ρ, acc)
"""
function update_initialstate_mv!(Xvec,Xvecᵒ,Wvec,ll,x,xᵒ,∇x, ∇xᵒ,result, resultᵒ,llout, lloutᵒ,
                sampler, Qvec, mask, mask_id, δ, acc)
    nshapes = length(Xvec)
    n = Qvec[1].target.n
    if obs_atzero
        δvec = repeat([ones(d);fill(δ[2],d)],n)
    else
        δvec = repeat([fill(δ[1],d);fill(δ[2],d)],n)
    end
    if sampler in [:sgd, :sgld] # ADJUST LATER
        sample!(W, Wiener{Vector{StateW}}())
        cfg = ForwardDiff.GradientConfig(slogρ(Q, W, X), x, ForwardDiff.Chunk{2*d*n}()) # 2*d*P.n is maximal
        ForwardDiff.gradient!(∇x, slogρ(Q, W, X),x,cfg) # X gets overwritten but does not change
        if sampler==:sgd
            x .+= δ * mask .* ∇x
        end
        if sampler==:sgld
            x .+= .5*δ*mask.*∇x + sqrt(δ)*mask.*randn(2d*Q.target.n)
        end
        obj = simguidedlm_llikelihood!(LeftRule(), X, deepvec2state(x), W, Q; skip=sk)
        println("obj ", obj)
    end
    if sampler==:mcmc
        cfg = ForwardDiff.GradientConfig(slogρ_mv!(Qvec, Wvec, Xvec,llout), x, ForwardDiff.Chunk{2*d*n}()) # 2*d*P.n is maximal
        ForwardDiff.gradient!(result, slogρ_mv!(Qvec, Wvec, Xvec,llout),x,cfg) # X gets overwritten but does not change
        ll_incl0 = sum(llout) #ll_incl0 = DiffResults.value(result)
        ∇x .=  DiffResults.gradient(result)

        xᵒ .= x .+ .5 * δvec .* mask.* ∇x .+ sqrt.(δvec) .* mask .* randn(length(x))
        cfgᵒ = ForwardDiff.GradientConfig(slogρ_mv!(Qvec, Wvec, Xvec,lloutᵒ), xᵒ, ForwardDiff.Chunk{2*d*n}()) # 2*d*P.n is maximal
        ForwardDiff.gradient!(resultᵒ, slogρ_mv!(Qvec, Wvec, Xvecᵒ,lloutᵒ),xᵒ,cfgᵒ) # Xvecᵒ gets overwritten but does not change
        ll_incl0ᵒ = sum(lloutᵒ)
        ∇xᵒ .=  DiffResults.gradient(resultᵒ)

        # xstate = deepvec2state(x)
        # xᵒstate = deepvec2state(xᵒ)
        dn = sum(mask_id.>0)
        ndistr = MvNormal(zeros(dn),sqrt.(δvec)[mask_id])
        accinit = ll_incl0ᵒ - ll_incl0
                 - logpdf(ndistr,(xᵒ - x - .5*δvec .* mask.* ∇x)[mask_id]) +
                logpdf(ndistr,(x - xᵒ - .5*δvec .* mask.* ∇xᵒ)[mask_id])

        # compute acc prob
        if log(rand()) <= accinit
            x .= xᵒ
            for k in 1:nshapes
                for i in eachindex(Xvec[k].yy)
                    Xvec[k].yy[i] .= Xvecᵒ[k].yy[i]
                end
            end
            println("update initial state; accinit: ", accinit, "  accepted")
            acc[2] +=1
            obj = ll_incl0ᵒ
            ll .= lloutᵒ
        else
            println("update initial state; accinit: ", accinit, "  rejected")
            obj = ll_incl0
            ll .= llout
        end
    end
    obj
#    x, Xvec, ll, obj, acc
end


function update_pars(P, tt_, mT, guidrecvecᵒ, (LT,ΣT,μT), (L0, Σ0),
            Xvec, Xvecᵒ,Wvec, Qvec, Qvecᵒ, x, ll, (prior_a, prior_c, prior_γ), (σ_a,σ_c,σ_γ), acc)
    nshapes = length(Xvec)
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
    Qvecᵒ .= [construct_guidedproposal!(tt_, guidrecvecᵒ[k], (LT,ΣT,μT), (L0, Σ0),
        (Qvec[k].xobs0, Qvec[k].xobsT), Pᵒ, Pauxvecᵒ[k]) for k in 1:nshapes]
    llᵒ = simguidedlm_llikelihood!(LeftRule(), Xvecᵒ, deepvec2state(x), Wvec, Qvecᵒ; skip=sk)
    A = logpdf(prior_a,aᵒ) - logpdf(prior_a,P.a) +
            logpdf(prior_c,cᵒ) - logpdf(prior_c,P.c) +
            logpdf(prior_γ,γᵒ) - logpdf(prior_γ,getγ(P)) +
                sum(llᵒ) - sum(ll) +
                logpdf(LogNormal(log(Pᵒ.a),σ_a),P.a)- logpdf(LogNormal(log(P.a),σ_a),Pᵒ.a)+
                logpdf(LogNormal(log(Pᵒ.c),σ_c),P.c)- logpdf(LogNormal(log(P.c),σ_c),Pᵒ.c)+
                logpdf(LogNormal(log(getγ(Pᵒ)),σ_γ),getγ(P))- logpdf(LogNormal(log(getγ(P)),σ_γ),getγ(Pᵒ))

    print("logaccept for parameter update ", round(A;digits=4))
    if log(rand()) <= A  # assume symmetric proposal and uniform prior, adjust later
        print("  accepted")
         #P, Pᵒ = Pᵒ, P
        # Xvec, Xvecᵒ = Xvecᵒ, Xvec
        # Pauxvec, Pauxvecᵒ = Pauxvecᵒ, Pauxvec
        # Qvec, Qvecᵒ = Qvecᵒ, Qvec
        # ll, llᵒ = llᵒ, ll
         P = Pᵒ
         Xvec .= Xvecᵒ
         Qvec .= Qvecᵒ
         ll .= llᵒ
        acc[3] +=1
    else
                print("  rejected")
    end
    P, acc
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
        (δ, prior_a, prior_c, prior_γ, σ_a, σ_c, σ_γ),
        outdir, pb; updatepars = true, makefig=true, showmomenta=false)

    StateW = PointF
    dwiener = dimwiener(P)
    if obs_atzero
        L0 = LT = [(i==j)*one(UncF) for i in 1:2:2P.n, j in 1:2P.n]
        Σ0 = ΣT = [(i==j)*σobs^2*one(UncF) for i in 1:P.n, j in 1:P.n]
        mask = deepvec(State(0 .- 0*xinit.q, 1 .- 0*(xinit.p)))  # only optimize momenta
    else
        L0 = Array{UncF}(undef,0,2*P.n)
        Σ0 = Array{UncF}(undef,0,0)
        xobs0 = Array{PointF}(undef,0)
        LT = [(i==j)*one(UncF) for i in 1:2:2P.n, j in 1:2P.n]
        ΣT = [(i==j)*σobs^2*one(UncF) for i in 1:P.n, j in 1:P.n]
        mask = deepvec(State(1 .- 0*xinit.q, 1 .- 0*(xinit.p)))  # only optimize positions and momenta
    end
    μT = zeros(PointF,P.n)
    mask_id = (mask .> 0.1) # get indices that correspond to momenta

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
    acc = zeros(3) # keep track of mcmc accept probs (first comp is for CN update; 2nd component for langevin update on initial momenta, 3rd parameter updates)
    Xsave = typeof(zeros(length(tt_) * P.n * 2 * d))[]
    parsave = Vector{Float64}[]
    push!(Xsave, convert_samplepath(Xvec[1]))
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
    result = DiffResults.GradientResult(x)
    resultᵒ = DiffResults.GradientResult(xᵒ)
    llout = copy(ll)
    lloutᵒ = copy(ll)

    if makefig
        #plotlandmarkpositions(X,P,xobs0,xobsT;db=4)
        xobs0comp1 = extractcomp(xobs0,1)
        xobs0comp2 = extractcomp(xobs0,2)
        xobsTcomp1 = extractcomp(xobsTvec[1],1)
        xobsTcomp2 = extractcomp(xobsTvec[1],2)
        pp1 = plotshapes(xobs0comp1,xobs0comp2,xobsTcomp1, xobsTcomp2)
    end

#    for i in 1:ITER
    anim =    @animate for i in 1:ITER
        if makefig
            drawpath(i-1,P.n,x,Xvec[1],objvals,parsave,(xobs0comp1,xobs0comp2,xobsTcomp1, xobsTcomp2),pb)
        end
        println("iteration $i")

        # updates paths
        update_path!(Xvec, Xvecᵒ, Wvec, Wᵒ, Wnew, ll, x, sampler, Qvec, ρ, acc)

        # update initial state
        obj = update_initialstate_mv!(Xvec,Xvecᵒ,Wvec,ll,x,xᵒ,∇x, ∇xᵒ,result, resultᵒ,llout, lloutᵒ,
                            sampler, Qvec, mask, mask_id, δ, acc)

#print(Xvec[1].yy[1])

        # update parameters
        P, acc= update_pars(P, tt_, mT, guidrecvecᵒ, (LT,ΣT,μT), (L0, Σ0),
                    Xvec, Xvecᵒ,Wvec, Qvec, Qvecᵒ, x, ll, (prior_a, prior_c, prior_γ), (σ_a,σ_c,σ_γ), acc)

        println()
        # save some of the results
        if i in subsamples
            push!(Xsave, convert_samplepath(Xvec[1]))
            push!(parsave, [P.a, P.c, getγ(P)])
            push!(objvals, obj)
        end
        #
        if makefig && (i==ITER)
            drawpath(ITER,P.n,x,Xvec[1],objvals,parsave,(xobs0comp1,xobs0comp2,xobsTcomp1, xobsTcomp2),pb)
        end
    end
    perc_acc = 100acc/(nshapes*ITER)
    println("Acceptance percentages (bridgepath - inital state - parameters): ",perc_acc)
    anim, Xsave, parsave, objvals, perc_acc
end
