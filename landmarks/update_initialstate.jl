"""
    update initial state

    x , W, X, ll, obj, acc = update_initialstate!(Xvec,Xvecᵒ,Wvec,ll,x,xᵒ,∇x, ∇xᵒ,llout, lloutᵒ,
                    sampler, Qvec, δ, acc,updatekernel)

    #    updatekernel can be :mala_pos, :mala_mom, :mala_posandmom, :lmforward_pos

"""
function update_initialstate!(Xvec,Xvecᵒ,Wvec,ll,x,xᵒ,∇x, ∇xᵒ,llout, lloutᵒ,
                sampler, Qvec, δ, acc,updatekernel,ptemp, iter)
    nshapes = length(Xvec)
    n = Qvec[1].target.n
    x0 = deepvec2state(x)
    P = Qvec[1].target

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
        accinit = ll_incl0 = ll_incl0ᵒ = 0.0 # define because of scoping rules
                #b_grad = 1000.0
                #Dx = b_grad * ∇x / max(b_grad,norm(∇x))
                #xᵒ .= x .+ .5 * δvec .* mask.* Dx .+ sqrt.(δvec) .* mask .* randn(length(x))
        if updatekernel in [:mala_pos, :mala_mom]
            cfg = ForwardDiff.GradientConfig(slogρ!(Qvec, Wvec, Xvec,llout), x, ForwardDiff.Chunk{2*d*n}()) # 2*d*P.n is maximal
            ForwardDiff.gradient!(∇x, slogρ!(Qvec, Wvec, Xvec,llout),x,cfg) # X gets overwritten but does not change
            ll_incl0 = sum(llout)
            if updatekernel==:mala_pos
                mask = deepvec(State(1 .- 0*x0.q,  0*x0.p))
                stepsize = δ[1]
            elseif updatekernel==:mala_mom
                mask = deepvec(State(0*x0.q, 1 .- 0*x0.p))
                stepsize = δ[2]
            end
            mask_id = (mask .> 0.1) # get indices that correspond to positions or momenta
            xᵒ .= x .+ .5 * stepsize * mask.* ∇x .+ sqrt(stepsize) .* mask .* randn(length(x))                              # should be ".=" or just "="?
            cfgᵒ = ForwardDiff.GradientConfig(slogρ!(Qvec, Wvec, Xvecᵒ,lloutᵒ), xᵒ, ForwardDiff.Chunk{2*d*n}()) # 2*d*P.n is maximal
            ForwardDiff.gradient!(∇xᵒ, slogρ!(Qvec, Wvec, Xvecᵒ,lloutᵒ),xᵒ,cfgᵒ) # Xvecᵒ gets overwritten but does not change
            ll_incl0ᵒ = sum(lloutᵒ)
            ndistr = MvNormal(d * n,sqrt(stepsize))
            accinit = ll_incl0ᵒ - ll_incl0 -
                      logpdf(ndistr,(xᵒ - x - .5*stepsize .* mask.* ∇x)[mask_id]) +
                     logpdf(ndistr,(x - xᵒ - .5*stepsize .* mask.* ∇xᵒ)[mask_id])
        elseif updatekernel==:mala_posandmom
            δvec = repeat([fill(δ[1],d);fill(δ[2],d)],n)
            cfg = ForwardDiff.GradientConfig(slogρ!(Qvec, Wvec, Xvec,llout), x, ForwardDiff.Chunk{2*d*n}()) # 2*d*P.n is maximal
            ForwardDiff.gradient!(∇x, slogρ!(Qvec, Wvec, Xvec,llout),x,cfg) # X gets overwritten but does not change
            ll_incl0 = sum(llout)
            xᵒ .= x .+ .5 .* δvec .* ∇x .+ sqrt.(δvec) .* randn(length(x))                              # should be ".=" or just "="?
            cfgᵒ = ForwardDiff.GradientConfig(slogρ!(Qvec, Wvec, Xvecᵒ,lloutᵒ), xᵒ, ForwardDiff.Chunk{2*d*n}()) # 2*d*P.n is maximal
            ForwardDiff.gradient!(∇xᵒ, slogρ!(Qvec, Wvec, Xvecᵒ,lloutᵒ),xᵒ,cfgᵒ) # Xvecᵒ gets overwritten but does not change
            ll_incl0ᵒ = sum(lloutᵒ)
            dn = length(δvec)
            ndistr = MvNormal(diagm(0=>δvec))
            accinit = ll_incl0ᵒ - ll_incl0 -
                       logpdf(ndistr,(xᵒ - x - .5 * δvec .* ∇x)) +
                      logpdf(ndistr,(x - xᵒ - .5 * δvec .* ∇xᵒ))
            # plotting
            Pdeterm = MarslandShardlow(0.1, 0.1, 0.0, 0.0, P.n)
            plotlandmarkpositions(initSamplePath(0:0.01:0.1,x0),Pdeterm,x0.q,deepvec2state(xᵒ).q;db=2.0)
        # elseif updatekernel==:amala
        #     nx = length(x)
        #     ϵ0 = 0.01
        #     δamala  = 0.001#0.01
        #     b_grad = 1.0
        #     Dx = b_grad * ∇x / max(b_grad,norm(∇x))
        #     Ndistr = MvNormal(δamala * ( Diagonal(fill(ϵ0,nx)) .+ Bridge.outer(Dx)))
        #     N = rand(Ndistr)
        #     xᵒ .= x .+ δamala * Dx .+ N
        #     cfgᵒ = ForwardDiff.GradientConfig(slogρ!(Qvec, Wvec, Xvecᵒ,lloutᵒ), xᵒ, ForwardDiff.Chunk{2*d*n}()) # 2*d*P.n is maximal
        #     ForwardDiff.gradient!(∇xᵒ, slogρ!(Qvec, Wvec, Xvecᵒ,lloutᵒ),xᵒ,cfgᵒ) # Xvecᵒ gets overwritten but does not change
        #     ll_incl0ᵒ = sum(lloutᵒ)
        #     Dxᵒ = b_grad * ∇xᵒ / max(b_grad,norm(∇xᵒ))
        #     Ndistrᵒ = MvNormal(δamala * ( Diagonal(fill(ϵ0,nx)) .+ Bridge.outer(Dxᵒ)))
        #
        #      accinit = ll_incl0ᵒ - ll_incl0 - logpdf(Ndistr,N) +
        #              logpdf(Ndistrᵒ,x - xᵒ - δamala * Dxᵒ)
        elseif updatekernel==:lmforward_pos
            Pdeterm = MarslandShardlow(0.1, 0.1, 0.0, 0.0, P.n)
            h = 0.1#1.0 #sqrt(κ)
            ∇xq = deepvec2state(∇x).q
            K = reshape([kernel(x0.q[i]- x0.q[j],Pdeterm) * one(UncF) for i in 1:P.n for j in 1:P.n], P.n, P.n)
            lcholK = lchol(K)
            zz = LinearAlgebra.naivesub!(lcholK',  randn(PointF, P.n))  #zz = randn(PointF, P.n)
            palg = 0.8
            bool_forward = rand() < palg
            if bool_forward
                ptemp = ∇xq + h * zz
            else
                ptemp = -∇xq - h * zz
            end
            xs = NState(x0.q, ptemp)
            nsteps = 1_00
            Δt = rand(Uniform(0.005, 0.01))
            hh = Δt/nsteps
            tsub = 0:hh:nsteps*hh                    #0:0.005:tsubend
            Wtemp = initSamplePath(tsub,  zeros(PointF, dimwiener(Pdeterm)))
            # forward simulate landmarks
            Xtemp = initSamplePath(tsub,xs)
            solve!(EulerMaruyama!(), Xtemp, xs, Wtemp, Pdeterm)
            #ptempᵒ = - Xtemp.yy[end].p

            xᵒState = NState(Xtemp.yy[end].q, x0.p)
            xᵒ .= deepvec(xᵒState)
            # cfg = ForwardDiff.GradientConfig(slogρ!(Qvec, Wvec, Xvecᵒ,lloutᵒ), xᵒ, ForwardDiff.Chunk{2*d*n}()) # 2*d*P.n is maximal
            # ForwardDiff.gradient!(∇xᵒ, slogρ!(Qvec, Wvec, Xvecᵒ,lloutᵒ),xᵒ,cfg)
            # ∇xqᵒ = deepvec2state(∇xᵒ).q
            #ptempᵒ = ptempᵒ - κ * (∇xpᵒ-ptempᵒ) - hh * zz

            #######################
            if false
                # check that going backwards with negative shadow momentum returns to x0.q
                Δt = 0.5; hh = Δt/nsteps;    tsub = 0:hh:nsteps*hh                    #0:0.005:tsubend
                Wtemp = initSamplePath(tsub,  zeros(PointF, dimwiener(Pdeterm)))
                Xtemp = initSamplePath(tsub,xs)

                hh = 1.0
                zz = LinearAlgebra.naivesub!(lcholK',  randn(PointF, P.n))
                ptemp = ∇xq +  hh * zz   # mala
                solve!(EulerMaruyama!(), Xtemp, NState(x0.q, ptemp), Wtemp, Pdeterm)
                ptempᵒ = - Xtemp.yy[end].p
                xᵒState = NState(Xtemp.yy[end].q, ptempᵒ)


                solve!(EulerMaruyama!(), Xtemp, xᵒState, Wtemp, Pdeterm)
                @show (x0.q .- Xtemp.yy[end].q)
                @show (x0.p .- Xtemp.yy[end].p)
            end
            ############################

            plotlandmarkpositions(Xtemp,Pdeterm,x0.q,xᵒState.q;db=2.0)

            lloutᵒ = simguidedlm_llikelihood!(LeftRule(), Xvecᵒ, xᵒState, Wvec, Qvec; skip=sk)
            ll_incl0 = sum(llout)
            ll_incl0ᵒ = sum(lloutᵒ)
            if bool_forward
                accinit = ll_incl0ᵒ - ll_incl0 + log((1-palg)/palg)
            else
                accinit = ll_incl0ᵒ - ll_incl0 - log((1-palg)/palg)
            end
            #+
                                #    (logϕ(ptempᵒ-∇xqᵒ) - logϕ(ptemp-∇xq))/h^2
                             #(logϕ(xᵒState.q, ptempᵒ, P)  - logϕ(x0.q, ptemp, P))/h^2


#            #+ logϕ(ptempᵒ) - logϕ(ptemp)  # difference of target Hamiltonians
            #println("logϕ(ptempᵒ-∇xqᵒ)  ", logϕ(ptempᵒ-∇xqᵒ))

            #println("logϕ(ptemp-∇xq)  ", logϕ(ptemp-∇xq))

        elseif updatekernel==:lmforward # simply forward simulate the deterministic system for a while, not using any gradient information
            Pdeterm = MarslandShardlow(0.1, 0.1, 0.0, 0.0, P.n)
            nsteps = 1_00
            Δt = rand(Uniform(0.005, 0.01))
            hh = Δt/nsteps
            tsub = 0:hh:nsteps*hh                    #0:0.005:tsubend
            Wtemp = initSamplePath(tsub,  zeros(PointF, dimwiener(Pdeterm)))
            Xtemp = initSamplePath(tsub,x0)
            solve!(EulerMaruyama!(), Xtemp, NState(x0.q,deepvec2state(∇x).q), Wtemp, Pdeterm)
            xᵒState = NState(Xtemp.yy[end].q, x0.p)
            xᵒ .= deepvec(xᵒState)
            # lloutᵒ = simguidedlm_llikelihood!(LeftRule(), Xvecᵒ, xᵒState, Wvec, Qvec; skip=sk)
            ll_incl0 = sum(llout)
            #ll_incl0ᵒ = sum(lloutᵒ)

            ll_incl0ᵒ = ll_incl0
            plotlandmarkpositions(Xtemp,Pdeterm,x0.q,xᵒState.q;db=2.5)





        elseif updatekernel==:lmforward_postest
            Pdeterm = MarslandShardlow(0.1, 0.1, 0.0, 0.0, P.n)
                # start test
            ∇xp = deepvec2state(∇x).p
            h = 1.0
            Pshadow = DeterMSshadow2(0.1, 0.1, 0.0, P.n, h, x0.p , ∇xp)
            nsteps = 1_00
            Δt = rand(Uniform(0.5, 1.1))
            println("Δt: ",Δt)
            hh = Δt/nsteps
            tsub = 0:hh:nsteps*hh                    #0:0.005:tsubend
            Wtemp = initSamplePath(tsub,  zeros(PointF, dimwiener(Pdeterm)))
            # forward simulate landmarks
            Xtemp = initSamplePath(tsub,x0)

            K = reshape([kernel(x0.q[i]- x0.q[j],Pdeterm) * one(UncF) for i in 1:P.n for j in 1:P.n], P.n, P.n)
            lcholK = lchol(K)
            zz = LinearAlgebra.naivesub!(lcholK',  randn(PointF, P.n))
            xhms0 = NState(x0.q,∇xp + sqrt(h) * zz) # where to start the hamilt state
            solve!(EulerMaruyama!(), Xtemp, xhms0, Wtemp, Pshadow)
            xhmsend = NState(Xtemp.yy[end].q, -Xtemp.yy[end].p)
            xᵒState = NState(xhmsend.q,x0.p)
            lloutᵒ = simguidedlm_llikelihood!(LeftRule(), Xvecᵒ, xᵒState, Wvec, Qvec; skip=sk)
            ll_incl0 = sum(llout)
            ll_incl0ᵒ = sum(lloutᵒ)
            accinit = ll_incl0ᵒ - ll_incl0 +
                             hamiltonian(xhms0,Pshadow) - hamiltonian(xhmsend,Pshadow)
            plotlandmarkpositions(Xtemp,Pdeterm,x0.q,xᵒState.q;db=2.0)
            # end test
        end
        # compute acc prob
        if log(rand()) <= accinit
            obj = ll_incl0ᵒ
            boolacc = true
            llout .= lloutᵒ
#            println("check within update_initialstate! ", ll_incl0ᵒ-sum(lloutᵒ))
            println("update initial state ", updatekernel, " accinit: ", accinit, "  accepted")
            for k in 1:nshapes
                for i in 1:length(Xvec[1].yy)
                    Xvec[k].yy[i] .= Xvecᵒ[k].yy[i]
                end
            end
            if updatekernel == :lmforward_pos
                #ptemp .= ptempᵒ
            end
            accepted = 1
        else
            println("update initial state ", updatekernel, " accinit: ", accinit, "  rejected")
            obj = ll_incl0

        #    println("check within update_initialstate! ", ll_incl0-sum(llout))
            boolacc = false
            accepted = 0
        end
    end
    boolacc, obj, (kernel = updatekernel, acc = accepted), xᵒ, ∇xᵒ, llout
end


logϕ(p) = -0.5 * norm(p)^2
logϕ(qfix, p, P) = -hamiltonian(NState(qfix,p),P)
function hamiltonian(x::NState, P::MarslandShardlow)
    s = 0.0
    for i in 1:P.n, j in 1:P.n
        s += dot(x.p[i], x.p[j])*kernel(x.q[i] - x.q[j], P)
    end
    0.5 * s
end
