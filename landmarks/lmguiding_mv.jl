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
struct GuidedProposalnew!{T,Ttarget,Taux,TL,Txobs0,TxobsT,F} <: ContinuousTimeProcess{T}
    target::Ttarget   # P
    aux::Taux      # Ptilde
    tt::Vector{Float64}  # grid of time points on single segment (S,T]
    guidrec::TL # recursions on grid tt
    xobs0::Txobs0   # observation at time 0
    xobsT::TxobsT  # observation at time T
    endpoint::F

    function GuidedProposalnew!(target, aux, tt_, guidrec, xobs0, xobsT,  endpoint=Bridge.endpoint)
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
function lmgpupdatenew!(Lt0₊, Mt⁺0₊::Array{Pnt,2}, μt0₊, (L0, Σ0, xobs0), Lt0, Mt⁺0, μt0) where Pnt
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
function construct_guidedproposalnew!(tt_, guidrec, (LT,ΣT,μT), (L0, Σ0), (xobs0, xobsT), P, Paux)
    Lt0₊, Mt⁺0₊, μt0₊ =  guidingbackwards!(Lm(), tt_, (guidrec.Lt, guidrec.Mt⁺,guidrec.μt), Paux, (LT, ΣT, μT))
    lmgpupdatenew!(Lt0₊, Mt⁺0₊, μt0₊, (L0, Σ0, xobs0),guidrec.Lt0, guidrec.Mt⁺0, guidrec.μt0)
    guidrec.Mt = map(X -> InverseCholesky(lchol(X)),guidrec.Mt⁺)
    for i in 1:length(tt_)
        guidrec.Ht[i] .= guidrec.Lt[i]' * (guidrec.Mt[i] * guidrec.Lt[i] )
    end
    GuidedProposalnew!(P, Paux, tt_, guidrec, xobs0, xobsT)
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

target(Q::GuidedProposalnew!) = Q.target
auxiliary(Q::GuidedProposalnew!) = Q.aux
constdiff(Q::GuidedProposalnew!) = constdiff(target(Q)) && constdiff(auxiliary(Q))

function _b!((i,t), x::State, out::State, Q::GuidedProposalnew!)
    Bridge.b!(t, x, out, Q.target)
    out .+= amul(t,x,Q.guidrec.Lt[i]' * (Q.guidrec.Mt[i] *(Q.xobsT-Q.guidrec.μt[i]-Q.guidrec.Lt[i]*vec(x))),Q.target)
    out
end

σ!(t, x, dw, out, Q::GuidedProposalnew!) = σ!(t, x, dw, out, Q.target)

function _r!((i,t), x::State, out::State, Q::GuidedProposalnew!)
    out .= vecofpoints2state(Q.guidrec.Lt[i]' * (Q.guidrec.Mt[i] *(Q.xobsT-Q.guidrec.μt[i]-Q.guidrec.Lt[i]*vec(x))))
    out
end
# need function that multiplies square unc with state and outputs state

function guidingterm((i,t),x,Q::GuidedProposalnew!)
    #Bridge.b(t,x,Q.target) +
    amul(t,x,Q.guidrec.Lt[i]' * (Q.guidrec.Mt[i] *(Q.xobsT-Q.guidrec.μt[i]-Q.guidrec.Lt[i]*vec(x))),Q.target)
end
"""
Returns the guiding terms a(t,x)*r̃(t,x) along the path of a guided proposal
for each value in X.tt.
Hence, it returns an Array of type State
"""
function guidingterms(X,Q::GuidedProposalnew!)
    i = first(1:length(X.tt))
    out = [guidingterm((i,X.tt[i]),X.yy[i],Q)]
    for i in 2:length(X.tt)
        push!(out, guidingterm((i,X.tt[i]),X.yy[i],Q))
    end
    out
end

function Bridge.lptilde(x, Q)
  y = deepvec([Q.xobs0; Q.xobsT] - Q.guidrec.μt0 - Q.guidrec.Lt0*vec(x))
  M⁺0deep = deepmat(Q.guidrec.Mt⁺0)
  -0.5*logdet(M⁺0deep) -0.5*dot(y, M⁺0deep\y)
end



"""
    Simulate guided proposal and compute loglikelihood

    solve sde inplace and return loglikelihood;
    thereby avoiding 'double' computations
"""
function simguidedlm_llikelihood!(::LeftRule,  Xᵒ, x0, W, Q::GuidedProposalnew!; skip = 0, ll0 = true)
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
        logρ0 = lptilde(x0,Q)
    else
        logρ0 = 0.0 # don't compute
    end
    copyto!(Xᵒ.yy[end], Bridge.endpoint(Xᵒ.yy[end],Q))
    som + logρ0
end


if TEST # first run lm_main.jl
    StateW = PointF
    dwiener = dimwiener(P)
    L0 = LT = [(i==j)*one(UncF) for i in 1:2:2P.n, j in 1:2P.n]
    Σ0 = ΣT = [(i==j)*σobs^2*one(UncF) for i in 1:P.n, j in 1:P.n]
    μT = zeros(PointF,P.n)

    # now the new stuff:
    nshapes = 2
    xobsTvec = [xobsT, 2*xobsT] # just a simple example
    guidres = init_guidrec(tt_, LT, ΣT, μT, L0, Σ0, xobs0)
    guidrecvec = [guidres for i in 1:nshapes]  # memory allocation for each shape
    Pauxvec = [auxiliary(P,State(xobsTvec[i],mT)) for i in 1:nshapes] # auxiliary process for each shape
    Qvec = [construct_guidedproposalnew!(tt_, guidrecvec[i], (LT,ΣT,μT), (L0, Σ0),
            (xobs0, xobsTvec[i]), P, Pauxvec[i]) for i in 1:nshapes]


    X = [initSamplePath(tt_, xinit) for i in 1:nshapes]
    W = [initSamplePath(tt_,  zeros(StateW, dwiener)) for i in 1:nshapes]
    for i in 1:nshapes
        sample!(W[i], Wiener{Vector{StateW}}())
    end
    ll = [simguidedlm_llikelihood!(LeftRule(), X[i], xinit, W[i], Qvec[i]; skip=sk) for i in 1:nshapes]

    guidrecvecᵒ = [init_guidrec(tt_,LT,ΣT,μT,Σ0, xobs0) for i in 1:nshapes]  # memory allocation for each shape
# from here can follow lm_mcmc.jl and try to adjust

    # what if no observations at time 0?
    L0 = Array{UncF}(undef,0,2*P.n)
    Σ0 = Array{UncF}(undef,0,0)
    xobs0 = Array{PointF}(undef,0)

    guidrec = init_guidrec(tt_, LT, ΣT, μT, L0, Σ0, xobs0)
    guidrecvec = [guidrec for i in 1:nshapes]  # memory allocation for each shape
    Qvec = [construct_guidedproposalnew!(tt_, guidrecvec[i], (LT,ΣT,μT), (L0, Σ0),
            (xobs0, xobsTvec[i]), P, Pauxvec[i]) for i in 1:nshapes]

end
