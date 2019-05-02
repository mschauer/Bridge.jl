# compute guiding term: backward ode
# compute likelihood of guided proposal

# not sure about this
function lmgpupdate(Lt, Mt, μt, Σ, L, v)
    Lt = [Lt; L]
    Mt = [Σ 0I; 0I Mt]
    μt = [zero(size(L,2)); μt]
end



import Bridge: kernelr3!, R3!, target, auxiliary, constdiff, llikelihood, _b!, B!, σ!, b!

"""
Construct guided proposal on a single segment with times in tt from precomputed ν and H
"""
struct GuidedProposall!{T,Ttarget,Taux,TL,TM,Tμ,Txobs,F} <: ContinuousTimeProcess{T}
    target::Ttarget   # P
    aux::Taux      # Ptilde
    tt::Vector{Float64}  # grid of time points on single segment (S,T]
    Lt::Vector{TL}
    Mt::Vector{TM}
    μt::Vector{Tμ}
    xobs::Txobs
    endpoint::F

    function GuidedProposall!(target, aux, tt_, L, M, μ, xobs, endpoint=Bridge.endpoint)
        tt = collect(tt_)
        new{Bridge.valtype(target),typeof(target),typeof(aux),eltype(L),eltype(M),eltype(μt),typeof(xobs),typeof(endpoint)}(target, aux, tt, L, M, μ, xobs, endpoint)
    end
end


struct Lm  end

function guidingbackwards!(::Lm, t, (Lt, Mt⁺, μt), Paux, (Lend, Mend⁺, μend))
    Mt⁺[end], Lt[end], μt[end] = Σend, Lend, μend
    BB = deepmat(Bridge.B(0, Paux)) # does not depend on time
    aa = deepmat(Bridge.a(0, Paux)) # does not depend on time
    β = deepvec(Bridge.β(0,Paux)) # does not depend on time
    for i in length(t)-1:-1:1
        dt = t[i+1]-t[i]
        Lt[i] .=  Lt[i+1] * (I + BB * dt)
        Mt⁺[i] .= Mt⁺[i+1] + Lt[i+1]* aa * Matrix(Lt[i+1]') * dt
        μt[i] .=  μt[i+1] + Lt[i+1] * β * dt
    end
    (Lt[1], Mt⁺[1], μt[1])
end

target(Q::GuidedProposall!) = Q.target
auxiliary(Q::GuidedProposall!) = Q.aux

constdiff(Q::GuidedProposall!) = constdiff(target(Q)) && constdiff(auxiliary(Q))


function Bridge._b!((i,t), x::State, out::State, Q::GuidedProposall!)
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

function guidingterm((i,t),x::State,Q::GuidedProposall!)
    Bridge.b(t,x,Q.target) + amul(t,x,Q.Lt[i]' * (Q.Mt[i] *(Q.xobs-Q.μt[i]-Q.Lt[i]*vec(x))),Q.target)
end
"""
Returns the guiding terms a(t,x)*r̃(t,x) along the path of a guided proposal
"""
function guidingterms(X::SamplePath{State{SArray{Tuple{2},Float64,1,2}}},Q::GuidedProposall!)
    out = State[]
    for i in 1:length(X.tt)
        push!(out, guidingterm((i,X.tt[i]),X.yy[i],Q))
    end
    out
end


function llikelihood(::LeftRule, Xcirc::SamplePath, Q::GuidedProposall!; skip = 0)
    tt = Xcirc.tt
    xx = Xcirc.yy

    som::Float64 = 0.
    rout = copy(xx[1])
    bout = copy(rout)
    btout = copy(rout)
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
            H = H((i,s), x, Q)
            Δa =  a((i,s), x, target(Q)) - a((i,s), x, auxiliary(Q))
            som -= 0.5*tr(Δa*H) * dt
            som += 0.5*(rout'*Δa*rout) * dt
        end
    end
    som
end
