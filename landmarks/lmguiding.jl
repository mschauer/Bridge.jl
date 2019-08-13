import Bridge: kernelr3!, R3!, target, auxiliary, constdiff, llikelihood, _b!, B!, σ!, b!

"""
    Guided proposal update for newly incoming observation.
    The existing tuple (Lt, Mt, μt, xobst) is updated using
    Σ: covariance matrix of the incoming observation
    L: specification that L x is observed (where x is the 'full' state)
    newobs: new observation (obtained as L x + N(0,Σ))
"""
function lmgpupdate(Lt0₊, Mt⁺0₊::Array{Pnt,2}, μt0₊, (L0, Σ0, xobs0)) where Pnt
    Lt0 = [L0; Lt0₊]
    m = size(Σ0)[1]
    n = size(Mt⁺0₊)[2]
    Mt⁺0 = [Σ0 zeros(Pnt,m,n); zeros(Pnt,n,m) Mt⁺0₊]
    μt0 = [0*xobs0; μt0₊]
    Lt0, Mt⁺0, μt0
end

"""
    Initialise arrays for (L,M,μ) where each value is copied length(t) times
"""
function initLMμH(t,(L,M,μ))
    Lt =  [copy(L) for s in t]
    Mt⁺ = [copy(M) for s in t]
    μt = [copy(μ) for s in t]
    H = L' * (M * L )
    Ht = [copy(H) for s in t]
    Lt, Mt⁺ , μt, Ht
end

"""
Construct guided proposal on a single segment with times in tt from precomputed ν and H
"""
struct GuidedProposall!{T,Ttarget,Taux,TL,TM,Tμ,TH,Txobs0,TxobsT,TLt0,TMt⁺0,Tμt0,F} <: ContinuousTimeProcess{T}
    target::Ttarget   # P
    aux::Taux      # Ptilde
    tt::Vector{Float64}  # grid of time points on single segment (S,T]
    Lt::Vector{TL} # Lt on grid tt
    Mt::Vector{TM}  # Mt on grid tt
    μt::Vector{Tμ}  # μt on grid tt
    Ht::Vector{TH}  # Ht on grid tt
    xobs0::Txobs0   # observation at time 0
    xobsT::TxobsT  # observation at time T
    Lt0::TLt0      # Lt at time 0, after gpupdate step incorporating observation xobs0
    Mt⁺0::TMt⁺0    # inv(Mt) at time 0, after gpupdate step incorporating observation xobs0
    μt0::Tμt0       # μt at time 0, after gpupdate step incorporating observation xobs0
    endpoint::F

    function GuidedProposall!(target, aux, tt_, L, M, μ, H, xobs0, xobsT, Lt0,Mt⁺0,μt0, endpoint=Bridge.endpoint)
        tt = collect(tt_)
        new{Bridge.valtype(target),typeof(target),typeof(aux),eltype(L),eltype(M),eltype(μ),eltype(H),typeof(xobs0),typeof(xobsT),typeof(Lt0),typeof(Mt⁺0),typeof(μt0),typeof(endpoint)}(target, aux, tt, L, M, μ, H, xobs0, xobsT, Lt0, Mt⁺0,μt0, endpoint)
    end
end


struct Lm  end

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

target(Q::GuidedProposall!) = Q.target
auxiliary(Q::GuidedProposall!) = Q.aux
constdiff(Q::GuidedProposall!) = constdiff(target(Q)) && constdiff(auxiliary(Q))

function _b!((i,t), x::State, out::State, Q::GuidedProposall!)
    Bridge.b!(t, x, out, Q.target)
    out .+= amul(t,x,Q.Lt[i]' * (Q.Mt[i] *(Q.xobsT-Q.μt[i]-Q.Lt[i]*vec(x))),Q.target)
    out
end

σ!(t, x, dw, out, Q::GuidedProposall!) = σ!(t, x, dw, out, Q.target)

function _r!((i,t), x::State, out::State, Q::GuidedProposall!)
    out .= vecofpoints2state(Q.Lt[i]' * (Q.Mt[i] *(Q.xobsT-Q.μt[i]-Q.Lt[i]*vec(x))))
    out
end
# need function that multiplies square unc with state and outputs state

function guidingterm((i,t),x,Q::GuidedProposall!)
    #Bridge.b(t,x,Q.target) +
    amul(t,x,Q.Lt[i]' * (Q.Mt[i] *(Q.xobsT-Q.μt[i]-Q.Lt[i]*vec(x))),Q.target)
end
"""
Returns the guiding terms a(t,x)*r̃(t,x) along the path of a guided proposal
for each value in X.tt.
Hence, it returns an Array of type State
"""
function guidingterms(X,Q::GuidedProposall!)
    i = first(1:length(X.tt))
    out = [guidingterm((i,X.tt[i]),X.yy[i],Q)]
    for i in 2:length(X.tt)
        push!(out, guidingterm((i,X.tt[i]),X.yy[i],Q))
    end
    out
end

function Bridge.lptilde(x, Q)
  y = deepvec([Q.xobs0; Q.xobsT] - Q.μt0 - Q.Lt0*vec(x))
  M⁺0deep = deepmat(Q.Mt⁺0)
  -0.5*logdet(M⁺0deep) -0.5*dot(y, M⁺0deep\y)
end


#function llikelihood(::LeftRule, Xcirc::SamplePath{State{Pnt}}, Q::GuidedProposall!; skip = 0) where {Pnt}
function llikelihood(::LeftRule,  Xᵒ, Q::GuidedProposall!; skip = 0)
    Pnt = eltype(Xᵒ.yy[1])
    tt =  Xᵒ.tt
    xx =  Xᵒ.yy
    som::deepeltype(xx[1])  = 0.

    # initialise objects to write into
    # srout and strout are vectors of Points
    if !constdiff(Q) # use sizetypes?  # must be target(Q).nfs
        srout = zeros(Pnt, length(Q.target.nfs))
        strout = zeros(Pnt, length(Q.target.nfs))
    end
    # rout, bout, btout are of type State
    rout = copy(xx[1])
    bout = copy(rout)
    btout = copy(rout)

    if !constdiff(Q)
        At = Bridge.a((1,0), xx[1], auxiliary(Q))
        A = zeros(Unc{deepeltype(xx[1])}, 2Q.target.n,2Q.target.n)
    end

    for i in 1:length(tt)-1-skip #skip last value, summing over n-1 elements
        _r!((i,tt[i]), xx[i], rout, Q)
        b!(tt[i], xx[i], bout, target(Q))
        _b!((i,tt[i]), xx[i], btout, auxiliary(Q))

#        btitilde!((s,i), x, btout, Q)
        dt = tt[i+1]-tt[i]
        som += dot(bout-btout, rout) * dt

        if !constdiff(Q)
            σt!(tt[i], xx[i], rout, srout, target(Q))      #  σ(t,x)' * tilder(t,x)
            σt!(tt[i], xx[i], rout, strout, auxiliary(Q))  #  tildeσ(t,x)' * tilder(t,x)

            som += 0.5*Bridge.inner(srout) * dt    # |σ(t,x)' * tilder(t,x)|^2
            som -= 0.5*Bridge.inner(strout) * dt   # |tildeσ(t,x)' * tilder(t,x)|^2

            Bridge.a!((i,tt[i]), xx[i], A, target(Q))  #A = Bridge.a((i,s), x, target(Q))

            som += 0.5*(dot(At,Q.Ht[i]) - dot(A,Q.Ht[i])) * dt
        end

    end
    som
end



function construct_guidedproposal!(tt_, (Lt, Mt⁺ , μt, Ht), (LT,ΣT,μT), (L0, Σ0), (xobs0, xobsT), P, Paux)
    (Lt0₊, Mt⁺0₊, μt0₊) =  guidingbackwards!(Lm(), tt_, (Lt, Mt⁺,μt), Paux, (LT, ΣT, μT))
    Lt0, Mt⁺0, μt0 = lmgpupdate(Lt0₊, Mt⁺0₊, μt0₊, (L0, Σ0, xobs0))
    Mt = map(X -> InverseCholesky(lchol(X)),Mt⁺)
    for i in 1:length(tt_)
        Ht[i] .= Lt[i]' * (Mt[i] * Lt[i] )
    end
    GuidedProposall!(P, Paux, tt_, Lt, Mt, μt, Ht, xobs0, xobsT, Lt0, Mt⁺0, μt0)
end


"""
    Simulate guided proposal and compute loglikelihood

    solve sde inplace and return loglikelihood;
    thereby avoiding 'double' computations
"""
function simguidedlm_llikelihood!(::LeftRule,  Xᵒ, x0, W, Q::GuidedProposall!; skip = 0, ll0 = true)
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
                som += 0.5*(dot(At,Q.Ht[i]) - dot(A,Q.Ht[i])) * dt
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

if TEST
    Y = deepcopy(Xᵒ)
    Ynew = deepcopy(Xᵒ)
    @time Bridge.solve!(EulerMaruyama!(), Y, xinit, Wᵒ, Q)
    @time llikelihood(LeftRule(), Y, Q; skip = 1)
    @time simguidedlm_llikelihood!(LeftRule(), Ynew, xinit, Wᵒ, Q)
    j=30;print(Y.yy[j]-Ynew.yy[j])
end
