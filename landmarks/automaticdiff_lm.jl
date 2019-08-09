# convert dual to float, while retaining float if type is float
deepvalue(x::Float64) = x
deepvalue(x::ForwardDiff.Dual) = ForwardDiff.value(x)
deepvalue(x) = deepvalue.(x)
function deepvalue(x::State)
    State(deepvalue.(x.x))
end


function slogρ(x0deepv, Q, W,X) # stochastic approx to log(ρ)
    x0 = deepvec2state(x0deepv)
    simguidedlm_llikelihood!(LeftRule(), X, x0, W, Q; skip=sk)
end
slogρ(Q, W, X) = (x) -> slogρ(x, Q, W,X)
∇slogρ(Q, W, X) = (x) -> ForwardDiff.gradient(slogρ(Q, W,X), x)


"""
    update initial momenta and/or guided proposals using either
    sgd, sgld or mcmc
"""
function updatepath!(X,Xᵒ,W,Wᵒ,Wnew,ll,x,xᵒ,∇x, ∇xᵒ,result, resultᵒ,
                sampler, Q,mask, mask_id, δ, ρ, acc)
    P = Q.target
    if sampler in [:sgd, :sgld]
        sample!(W, Wiener{Vector{StateW}}())
        cfg = ForwardDiff.GradientConfig(slogρ(Q, W, X), x, ForwardDiff.Chunk{2*d*P.n}()) # 2*d*P.n is maximal
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
        # From current state (x,W) with loglikelihood ll, update to (xᵒ, Wᵒ)

        # update W for fixed x
        sample!(Wnew, Wiener{Vector{PointF}}())
        Wᵒ.yy .= ρ * W.yy + sqrt(1-ρ^2) * Wnew.yy
        llᵒ = simguidedlm_llikelihood!(LeftRule(), Xᵒ, deepvec2state(x), Wᵒ, Q;skip=sk)
        if log(rand()) <= llᵒ - ll
            for i in eachindex(X.yy)
                X.yy[i] .= Xᵒ.yy[i]
                W.yy[i] .= Wᵒ.yy[i]
            end
            println("update innovation: ll $ll $llᵒ, diff_ll: ",round(llᵒ-ll;digits=3),"  accepted")
            ll = llᵒ
            acc[1] +=1
        else
            println("update innovation: ll $ll $llᵒ, diff_ll: ",round(llᵒ-ll;digits=3),"  rejected")
        end

        # MALA step (update x, for fixed W)
        cfg = ForwardDiff.GradientConfig(slogρ(Q, W, X), x, ForwardDiff.Chunk{2*d*P.n}()) # 2*d*P.n is maximal
        ForwardDiff.gradient!(result, slogρ(Q, W, X),x,cfg) # X gets overwritten but does not change
        ll_incl0 = DiffResults.value(result)
        ∇x .=  DiffResults.gradient(result)

        xᵒ .= x .+ .5*δ * mask.* (∇x .+ sqrt(δ) * randn(length(x)))
        cfgᵒ = ForwardDiff.GradientConfig(slogρ(Q, W, Xᵒ), xᵒ, ForwardDiff.Chunk{2*d*P.n}()) # 2*d*P.n is maximal
        ForwardDiff.gradient!(resultᵒ, slogρ(Q, W, Xᵒ),xᵒ,cfgᵒ) # X gets overwritten but does not change
        ll_incl0ᵒ = DiffResults.value(resultᵒ)
        ∇xᵒ =  DiffResults.gradient(resultᵒ)

        xstate = deepvec2state(x)
        xᵒstate = deepvec2state(xᵒ)
        accinit = ll_incl0ᵒ - ll_incl0
                 - logpdf(MvNormal(d*P.n,sqrt(δ)),(xᵒ - x - .5*δ* mask.* ∇x)[mask_id]) +
                logpdf(MvNormal(d*P.n,sqrt(δ)),(x - xᵒ - .5*δ* mask.* ∇xᵒ)[mask_id])

        # compute acc prob
        if log(rand()) <= accinit
            x .= xᵒ
            for i in eachindex(X.yy)
                X.yy[i] .= Xᵒ.yy[i]
            end
            println("update initial state; accinit: ", accinit, "  accepted")
            acc[2] +=1
            obj = ll_incl0ᵒ
            ll = llᵒ
        else
            println("update initial state; accinit: ", accinit, "  rejected")
            obj = ll_incl0
        end
    end
    (x ,W, X), ll, obj, acc
end
