"""
Stochastic approximation to transition density.
Provide Wiener process.
"""
function slogpW(x0deepv, Lt0,  Mt⁺0, μt0, xobst0, Q, Wᵒ)
    x0 = deepvec2state(x0deepv)
    Xᵒ = Bridge.solve(EulerMaruyama!(), x0, Wᵒ, Q)# this causes the problem
    lptilde(x0, Lt0, Mt⁺0, μt0, xobst0) + llikelihood(LeftRule(), Xᵒ, Q; skip = 1)
end
slogpW(Lt0,  Mt⁺0, μt0, xobst0, Q, Wᵒ) = (x) -> slogpW(x, Lt0,  Mt⁺0, μt0, xobst0, Q, Wᵒ)
∇slogpW(Lt0,  Mt⁺0, μt0, xobst0, Q, Wᵒ) = (x) -> gradient(slogpW(Lt0,  Mt⁺0, μt0, xobst0, Q, Wᵒ), x)

function slogpWX(x0deepv, Lt0,  Mt⁺0, μt0, xobst0, Q, W,X) # preferred way
    x0 = deepvec2state(x0deepv)
#    solve!(EulerMaruyama!(), X, x0, W, Q)
# ll = llikelihood(LeftRule(), X, Q; skip = 1)
    X, ll = simguidedlm_llikelihood!(LeftRule(), X, x0, W, Q; skip=sk)
    lptilde(x0, Lt0, Mt⁺0, μt0, xobst0) + ll
end
slogpWX(Lt0,  Mt⁺0, μt0, xobst0, Q, Wᵒ,X) = (x) -> slogpWX(x, Lt0,  Mt⁺0, μt0, xobst0, Q, Wᵒ,X)
∇slogpWX(Lt0,  Mt⁺0, μt0, xobst0, Q, Wᵒ,X) = (x) -> gradient(slogpWX(Lt0,  Mt⁺0, μt0, xobst0, Q, Wᵒ,X), x)


"""
    update initial momenta and/or guided proposals using either
    sgd, sgld or mcmc
"""
function updatepath!(X,Xᵒ,W,Wᵒ,Wnew,ll,x,xᵒ,∇x, ∇xᵒ,
                sampler,(Lt0,  Mt⁺0, μt0, xobst0, Q),mask, mask_id, δ, ρ, acc)
    if sampler in [:sgd, :sgld]
        sample!(W, Wiener{Vector{StateW}}())
        # cfg = GradientConfig(slogpW(Lt0,  Mt⁺0, μt0, xobst0, Q, W), x, Chunk{d*P.n}()) # 2*d*P.n is maximal
        # @time gradient!(∇x, slogpW(Lt0,  Mt⁺0, μt0, xobst0, Q, W),x,cfg)
        cfg = GradientConfig(slogpWX(Lt0,  Mt⁺0, μt0, xobst0, Q, W, X), x, Chunk{2*d*P.n}()) # 2*d*P.n is maximal
        gradient!(∇x, slogpWX(Lt0,  Mt⁺0, μt0, xobst0, Q, W, X),x,cfg) # ensure X does not get overwritten


        if sampler==:sgd
            x .+= δ * mask.*∇x
        end
        if sampler==:sgld
            x .+= .5*δ*mask.*∇x + sqrt(δ)*mask.*randn(2d*Q.target.n)
        end
        xstate = deepvec2state(x)
        # Bridge.solve!(EulerMaruyama!(), X, xstate, W, Q)
        # obj = lptilde(xstate, Lt0, Mt⁺0, μt0, xobst0) +
        #         llikelihood(LeftRule(), X, Q; skip = 1)
        obj = lptilde(xstate, Lt0, Mt⁺0, μt0, xobst0) +
            simguidedlm_llikelihood!(LeftRule(), X, xstate, W, Q;skip=sk)
        println("obj ", obj)
    end
    if sampler==:mcmc
        # From current state (x,W) with loglikelihood ll, update to (xᵒ, Wᵒ)

        # update W for fixed x
        sample!(Wnew, Wiener{Vector{PointF}}())
        Wᵒ.yy .= ρ * W.yy + sqrt(1-ρ^2) * Wnew.yy
        # solve!(EulerMaruyama!(), Xᵒ, deepvec2state(x), Wᵒ, Q)
        # llᵒ = llikelihood(Bridge.LeftRule(), Xᵒ, Q,skip=sk)
        Xᵒ, llᵒ = simguidedlm_llikelihood!(LeftRule(), Xᵒ, deepvec2state(x), Wᵒ, Q;skip=sk)


        if log(rand()) <= llᵒ - ll

            for i in eachindex(X.yy)
                X.yy[i] .= Xᵒ.yy[i]
                W.yy[i] .= Wᵒ.yy[i]
            end
            #X, Xᵒ = Xᵒ, X
            #W, Wᵒ = Wᵒ, W

            ll = llᵒ
            println("update innovation: ll $ll $llᵒ, diff_ll: ",round(llᵒ-ll;digits=3),"  accepted")
            acc[1] +=1
        else
            println("update innovation: ll $ll $llᵒ, diff_ll: ",round(llᵒ-ll;digits=3),"  rejected")
        end

        # MALA step (update x, for fixed W)
        if !inplace
            ∇x .= ∇slogpW(Lt0,  Mt⁺0, μt0, xobst0, Q, W)(x)
        else
            #∇x .= ∇slogpWX(Lt0,  Mt⁺0, μt0, xobst0, Q, W, X)(x)
            cfg = GradientConfig(slogpWX(Lt0,  Mt⁺0, μt0, xobst0, Q, W, X), x, Chunk{2*d*P.n}()) # 2*d*P.n is maximal
            gradient!(∇x, slogpWX(Lt0,  Mt⁺0, μt0, xobst0, Q, W, X),x,cfg) # X gets overwritten but does not change
        end
        xᵒ .= x .+ .5*δ * mask.* (∇x .+ sqrt(δ) * randn(length(x)))
        if !inplace
             ∇xᵒ .= ∇slogpW(Lt0,  Mt⁺0, μt0, xobst0, Q, W)(xᵒ)
        else
             #∇xᵒ .= ∇slogpWX(Lt0,  Mt⁺0, μt0, xobst0, Q, W, X)(xᵒ)
             cfg = GradientConfig(slogpWX(Lt0,  Mt⁺0, μt0, xobst0, Q, W, Xᵒ), xᵒ, Chunk{2*d*P.n}()) # 2*d*P.n is maximal
             gradient!(∇xᵒ, slogpWX(Lt0,  Mt⁺0, μt0, xobst0, Q, W, Xᵒ),xᵒ,cfg) # Xᵒ gets overwritten and is changed
        end
        xstate = deepvec2state(x)
        xᵒstate = deepvec2state(xᵒ)
        # solve!(EulerMaruyama!(), Xᵒ, xᵒstate, W, Q)
        # ainit = lptilde(xᵒstate, Lt0, Mt⁺0, μt0, xobst0) + llikelihood(LeftRule(), Xᵒ, Q; skip = sk) -
        #     lptilde(xstate, Lt0, Mt⁺0, μt0, xobst0) - llikelihood(LeftRule(), X, Q; skip = sk) -
        #     logpdf(MvNormal(d*P.n,sqrt(δ)),(xᵒ - x - .5*δ* mask.* ∇x)[mask_id]) +
        #     logpdf(MvNormal(d*P.n,sqrt(δ)),(x - xᵒ - .5*δ* mask.* ∇xᵒ)[mask_id])

        lp = lptilde(xstate, Lt0, Mt⁺0, μt0, xobst0)
        #X, ll = simguidedlm_llikelihood!(LeftRule(), X, xstate, W, Q;skip=sk)
        #Xᵒ, llᵒ = simguidedlm_llikelihood!(LeftRule(), Xᵒ, xᵒstate, W, Q;skip=sk)
        llᵒ = llikelihood(LeftRule(), Xᵒ, Q; skip = sk)
        lpᵒ = lptilde(xᵒstate, Lt0, Mt⁺0, μt0, xobst0)

        ainit = lpᵒ + llᵒ - (lp + ll)
                 - logpdf(MvNormal(d*P.n,sqrt(δ)),(xᵒ - x - .5*δ* mask.* ∇x)[mask_id]) +
                logpdf(MvNormal(d*P.n,sqrt(δ)),(x - xᵒ - .5*δ* mask.* ∇xᵒ)[mask_id])

        # compute acc prob

        if log(rand()) <= ainit
            x .= xᵒ
            for i in eachindex(X.yy)
                X.yy[i] .= Xᵒ.yy[i]
            end
            println("update initial state; ainit: ", ainit, "  accepted")
            acc[2] +=1
            obj = lpᵒ + llᵒ
            ll = llᵒ
        else
            println("update initial state; ainit: ", ainit, "  rejected")
        #    ll = simguidedlm_llikelihood!(LeftRule(), X, xstate, W, Q;skip=sk)# just added
            obj = lp + ll
        end

    end
    #X,Xᵒ,W,Wᵒ,ll,x,xᵒ,∇x,∇xᵒ, obj,acc
    (x ,W, X), ll, obj, acc
end
