"""
Stochastic approximation to transition density.
Provide Wiener process.
"""
function slogpW(x0deepv, Lt0,  Mt⁺0, μt0, xobst0, Q, Wᵒ)
    x0 = deepvec2state(x0deepv)
    Xᵒ = Bridge.solve(EulerMaruyama!(), x0, Wᵒ, Q)# this causes the problem
    lptilde(vec(x0), Lt0, Mt⁺0, μt0, xobst0) + llikelihood(LeftRule(), Xᵒ, Q; skip = 1)
end
slogpW(Lt0,  Mt⁺0, μt0, xobst0, Q, Wᵒ) = (x) -> slogpW(x, Lt0,  Mt⁺0, μt0, xobst0, Q, Wᵒ)
∇slogpW(Lt0,  Mt⁺0, μt0, xobst0, Q, Wᵒ) = (x) -> gradient(slogpW(Lt0,  Mt⁺0, μt0, xobst0, Q, Wᵒ), x)

function slogpWX(x0deepv, Lt0,  Mt⁺0, μt0, xobst0, Q, Wᵒ,Xᵒ) # preferred way
    #dump(typeof(x0deepv))
    x0 = deepvec2state(x0deepv)
    solve!(EulerMaruyama!(), Xᵒ, x0, Wᵒ, Q)
    #dump(typeof(Xᵒ))
    lptilde(vec(x0), Lt0, Mt⁺0, μt0, xobst0) + llikelihood(LeftRule(), Xᵒ, Q; skip = 1)
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
        cfg = GradientConfig(slogpW(Lt0,  Mt⁺0, μt0, xobst0, Q, W), x, Chunk{d*P.n}()) # 2*d*P.n is maximal
        @time gradient!(∇x, slogpW(Lt0,  Mt⁺0, μt0, xobst0, Q, W),x,cfg)

        if sampler==:sgd
            x .+= δ*mask.*∇x
        end
        if sampler==:sgld
            x .+= .5*δ*mask.*∇x + sqrt(δ)*mask.*randn(2d*Q.target.n)
        end
        xstate = deepvec2state(x)
        Bridge.solve!(EulerMaruyama!(), X, xstate, W, Q)
        obj = lptilde(vec(xstate), Lt0, Mt⁺0, μt0, xobst0) +
                llikelihood(LeftRule(), X, Q; skip = 1)
        println("ll ", obj)
    end
    if sampler==:mcmc
        # Update W
        sample!(Wnew, Wiener{Vector{PointF}}())
        Wᵒ.yy .= ρ * W.yy + sqrt(1-ρ^2) * Wnew.yy
        solve!(EulerMaruyama!(), Xᵒ, deepvec2state(x), Wᵒ, Q)

        llᵒ = llikelihood(Bridge.LeftRule(), Xᵒ, Q,skip=sk)
        print("ll $ll $llᵒ, diff_ll: ",round(llᵒ-ll;digits=3))

        if log(rand()) <= llᵒ - ll
            X.yy .= Xᵒ.yy
            W.yy .= Wᵒ.yy
            ll = llᵒ
            print("update innovation accepted")
            acc[1] +=1
        else
            print("update innovation rejected")
        end
        println()

        # MALA step (update x)
        # ∇x  = rand(2*d*Q.target.n)
        # ∇xᵒ  = rand(2*d*Q.target.n)
        if !inplace
            ∇x .= ∇slogpW(Lt0,  Mt⁺0, μt0, xobst0, Q, W)(x)
        else
            #∇x .= ∇slogpWX(Lt0,  Mt⁺0, μt0, xobst0, Q, W, X)(x)
            cfg = GradientConfig(slogpWX(Lt0,  Mt⁺0, μt0, xobst0, Q, W, X), x, Chunk{2*d*P.n}()) # 2*d*P.n is maximal
            gradient!(∇x, slogpWX(Lt0,  Mt⁺0, μt0, xobst0, Q, W, X),x,cfg)
        end
        xᵒ .= x .+ .5*δ * mask.* (∇x .+ sqrt(δ) * randn(length(x)))
        if !inplace
             ∇xᵒ .= ∇slogpW(Lt0,  Mt⁺0, μt0, xobst0, Q, W)(xᵒ)
        else
             #∇xᵒ .= ∇slogpWX(Lt0,  Mt⁺0, μt0, xobst0, Q, W, X)(xᵒ)
             cfg = GradientConfig(slogpWX(Lt0,  Mt⁺0, μt0, xobst0, Q, W, X), xᵒ, Chunk{2*d*P.n}()) # 2*d*P.n is maximal
             gradient!(∇xᵒ, slogpWX(Lt0,  Mt⁺0, μt0, xobst0, Q, W, X),xᵒ,cfg)
        end
        xstate = deepvec2state(x)
        xᵒstate = deepvec2state(xᵒ)
        solve!(EulerMaruyama!(), Xᵒ, xᵒstate, W, Q)
        ainit = lptilde(vec(xᵒstate), Lt0, Mt⁺0, μt0, xobst0) + llikelihood(LeftRule(), Xᵒ, Q; skip = 1) -
            lptilde(vec(xstate), Lt0, Mt⁺0, μt0, xobst0) - llikelihood(LeftRule(), X, Q; skip = 1) -
            logpdf(MvNormal(d*P.n,sqrt(δ)),(xᵒ - x - .5*δ* mask.* ∇x)[mask_id]) +
            logpdf(MvNormal(d*P.n,sqrt(δ)),(x - xᵒ - .5*δ* mask.* ∇xᵒ)[mask_id])
        # compute acc prob
        print("ainit: ", ainit)
        if log(rand()) <= ainit
            x .= xᵒ
            xstate = xᵒstate
            X.yy .= Xᵒ.yy
            println("mala step accepted")
            acc[2] +=1
        else
            println("mala step rejected")
        end
        obj = lptilde(vec(xstate), Lt0, Mt⁺0, μt0, xobst0) +
                llikelihood(LeftRule(), X, Q; skip = 1)
    end
    X,Xᵒ,W,Wᵒ,ll,x,xᵒ,∇x,∇xᵒ, obj,acc
end
