import Bridge: kernelr3!, R3!, target, auxiliary, constdiff, llikelihood, _b!, B!, σ!, b!

"""
    Guided proposal update for newly incoming observation.
    The existing tuple (Lt, Mt, μt, xobst) is updated using
    Σ: covariance matrix of the incoming observation
    L: specification that L x is observed (where x is the 'full' state)
    newobs: new observation (obtained as L x + N(0,Σ))
"""
function lmgpupdate(Lt, Mt::Array{Pnt,2}, μt, xobst, (L, Σ, newobs)) where Pnt
    Lt = [L; Lt]
    m = size(Σ)[1]
    n = size(Mt)[2]
    Mt = [Σ zeros(Pnt,m,n); zeros(Pnt,n,m) Mt]
    μt = [0newobs; μt]
    xobst = [newobs; xobst]

    Lt, Mt, μt, xobst
end

"""
    Initialise arrays for (L,M,μ) where each value is copied length(t) times
"""
function initLMμ(t,(L,M,μ))
    Lt =  [copy(L) for s in t]
    Mt⁺ = [copy(M) for s in t]
    μt = [copy(μ) for s in t]
    Lt, Mt⁺ , μt
end

"""
Construct guided proposal on a single segment with times in tt from precomputed ν and H
"""
struct GuidedProposall!{T,Ttarget,Taux,TL,TM,Tμ,TH,Txobs,F} <: ContinuousTimeProcess{T}
    target::Ttarget   # P
    aux::Taux      # Ptilde
    tt::Vector{Float64}  # grid of time points on single segment (S,T]
    Lt::Vector{TL}
    Mt::Vector{TM}
    μt::Vector{Tμ}
    Ht::Vector{TH}
    xobs::Txobs
    endpoint::F

    function GuidedProposall!(target, aux, tt_, L, M, μ, H, xobs, endpoint=Bridge.endpoint)
        tt = collect(tt_)
        new{Bridge.valtype(target),typeof(target),typeof(aux),eltype(L),eltype(M),eltype(μt),eltype(H),typeof(xobs),typeof(endpoint)}(target, aux, tt, L, M, μ, H, xobs, endpoint)
    end
end


struct Lm  end


function guidingbackwards!(::Lm, t, (Lt, Mt⁺, μt), Paux, (L, Σ , μend))
    Mt⁺[end] .= Σ
    Lt[end] .= L
    BB = Matrix(Bridge.B(0, Paux)) # does not depend on time
    println("computing ã and its low rank approximation:")
    # various ways to compute ã (which does not depend on time);
    # low rank appoximation really makes sense here
#   @time    aa = Matrix(Bridge.a(0, Paux))        # vanilla, no lr approx
#   @time  aalr = pheigfact(deepmat(Matrix(Bridge.a(0, Paux))))      # low rank approx default
#   @time  aalr = pheigfact(deepmat(Matrix(Bridge.a(0, Paux))),rank=400)  # fix rank
    @time  aalr = pheigfact(deepmat(Matrix(Bridge.a(0, Paux))), rtol=1e-10)  # control accuracy of lr approx
    println("Rank ",size(aalr[:vectors],2), " approximation to ã")
    sqrt_aalr = deepmat2unc(aalr[:vectors] * diagm(0=> sqrt.(aalr[:values])))

    β = vec(Bridge.β(0,Paux)) # does not depend on time
    for i in length(t)-1:-1:1
        dt = t[i+1]-t[i]
#       Lt[i] .=  Lt[i+1] * (I + BB * dt)  # explicit
        Lt[i] .= Lt[i+1]/(I - dt* BB)  # implicit, similar computational cost
#       Mt⁺[i] .= Mt⁺[i+1] + Lt[i+1]* aa * Matrix(Lt[i+1]') * dt
        Mt⁺[i] .= Mt⁺[i+1] + Bridge.outer(Lt[i+1] * sqrt_aalr) * dt
        μt[i] .=  μt[i+1] + Lt[i+1] * β * dt
    end
    (Lt[1], Mt⁺[1], μt[1])
end

target(Q::GuidedProposall!) = Q.target
auxiliary(Q::GuidedProposall!) = Q.aux

constdiff(Q::GuidedProposall!) = constdiff(target(Q)) && constdiff(auxiliary(Q))


function _b!((i,t), x::State, out::State, Q::GuidedProposall!)
    Bridge.b!(t, x, out, Q.target)
    out .+= amul(t,x,Q.Lt[i]' * (Q.Mt[i] *(Q.xobs-Q.μt[i]-Q.Lt[i]*vec(x))),Q.target)
    out
end

σ!(t, x, dw, out, Q::GuidedProposall!) = σ!(t, x, dw, out, Q.target)

# in following x is of type state
function _r!((i,t), x::State, out::State, Q::GuidedProposall!)
    out .= vecofpoints2state(Q.Lt[i]' * (Q.Mt[i] *(Q.xobs-Q.μt[i]-Q.Lt[i]*vec(x))))
    out
end
# need function that multiplies square unc with state and outputs state

function guidingterm((i,t),x::State,Q::GuidedProposall!)
    #Bridge.b(t,x,Q.target) +
    amul(t,x,Q.Lt[i]' * (Q.Mt[i] *(Q.xobs-Q.μt[i]-Q.Lt[i]*vec(x))),Q.target)
end
"""
Returns the guiding terms a(t,x)*r̃(t,x) along the path of a guided proposal
for each value in X.tt.
Hence, it returns an Array of type State
"""
function guidingterms(X::SamplePath{State{SArray{Tuple{2},Float64,1,2}}},Q::GuidedProposall!)
    i = first(1:length(X.tt))
    out = [guidingterm((i,X.tt[i]),X.yy[i],Q)]
    for i in 2:length(X.tt)
        push!(out, guidingterm((i,X.tt[i]),X.yy[i],Q))
    end
    out
end

"""
v0 consists of all observation vectors stacked, so in case of two observations, it should be v0 and vT stacked
"""
function Bridge.lptilde(x, L0, M⁺0, μ0, xobs)
  y = deepvec(xobs - μ0 - L0*x)
  M⁺0deep = deepmat(M⁺0)
  -0.5*logdet(M⁺0deep) -0.5*dot(y, M⁺0deep\y)
end

function llikelihood(::LeftRule, Xcirc::SamplePath{State{Pnt}}, Q::GuidedProposall!; skip = 0) where {Pnt}
    tt = Xcirc.tt
    xx = Xcirc.yy
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

    At = Bridge.a((1,0), xx[1], auxiliary(Q))
    A = zeros(Unc{deepeltype(xx[1])}, 2Q.target.n,2Q.target.n)

    for i in 1:length(tt)-1-skip #skip last value, summing over n-1 elements
        s = tt[i]
        x = xx[i]
        _r!((i,s), x, rout, Q)
        b!(s, x, bout, target(Q))
        _b!((i,s), x, btout, auxiliary(Q))
#        btitilde!((s,i), x, btout, Q)
        dt = tt[i+1]-tt[i]
        som += dot(bout-btout, rout) * dt

        if !constdiff(Q)
            σt!(s, x, rout, srout, target(Q))      #  σ(t,x)' * tilder(t,x)
            σt!(s, x, rout, strout, auxiliary(Q))  #  tildeσ(t,x)' * tilder(t,x)

            som += 0.5*Bridge.inner(srout) * dt    # |σ(t,x)' * tilder(t,x)|^2
            som -= 0.5*Bridge.inner(strout) * dt   # |tildeσ(t,x)' * tilder(t,x)|^2

            Bridge.a!((i,s), x, A, target(Q))  #A = Bridge.a((i,s), x, target(Q))

            # som -= 0.5*hadamtrace(A, Q.Ht[i]) * dt
             # som += 0.5*hadamtrace(At, Q.Ht[i]) * dt
            som += 0.5*(dot(At,Q.Ht[i]) - dot(A,Q.Ht[i])) * dt
        end
    end
    som
end

construct_guidedproposal! = function(tt_, (Lt, Mt⁺ , μt), (LT,ΣT,μT), (L0, Σ0), (xobs0, xobsT), P, Paux)
    (Lt0₊, Mt⁺0₊, μt0₊) =  guidingbackwards!(Lm(), tt_, (Lt, Mt⁺,μt), Paux, (LT, ΣT, μT))
    Lt0, Mt⁺0, μt0, xobst0 = lmgpupdate(Lt0₊, Mt⁺0₊, μt0₊, xobsT, (L0, Σ0, xobs0))
    Mt = map(X -> InverseCholesky(lchol(X)),Mt⁺)
    Ht = [Lt[i]' * (Mt[i] * Lt[i] ) for i in 1:length(tt_) ]
    Q = GuidedProposall!(P, Paux, tt_, Lt, Mt, μt, Ht, xobsT)

    (Lt, Mt⁺ , μt), Q, (Lt0, Mt⁺0, μt0, xobst0)
end
