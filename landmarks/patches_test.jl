"""
    Simulate guided proposal and compute loglikelihood

    solve sde inplace and return loglikelihood;
    thereby avoiding 'double' computations
"""
# Fixme: do something with Bridge.endpoint
function simguidedlm_llikelihood!(::LeftRule,  Xᵒ, X0, W, Q::GuidedProposall!; skip = 0)
    Pnt = eltype(x0)
    tt =  Xᵒ.tt
    Xᵒ.yy[1] .= X0
    som::deepeltype(x0)  = 0.

    # initialise objects to write into
    # srout and strout are vectors of Points
    if isa(Q.target,Landmarks)
        srout = zeros(Pnt, length(Q.target.nfs))
        strout = zeros(Pnt, length(Q.target.nfs))
    end
    if isa(Q.target,MarslandShardlow)
        srout = zeros(Pnt, Q.target.n)
        strout = zeros(Pnt, Q.target.n)
    end
    rout = copy(X0)
    bout = copy(X0)
    btout = copy(X0)
    wout = copy(X0)

    if !constdiff(Q)
        At = Bridge.a((1,0), X0, auxiliary(Q))
        A = zeros(Unc{deepeltype(x0)}, 2Q.target.n,2Q.target.n)
    end

    for i in 1:length(tt)-1
        x = Xᵒ.yy[i]
        dt = tt[i+1]-tt[i]
        b!(tt[i], x, bout, target(Q)) # b(t,x)
        _r!((i,tt[i]), x, rout, Q) # tilder(t,x)
        σt!(tt[i], x, rout, srout, target(Q))      #  σ(t,x)' * tilder(t,x)
        Bridge.σ!(tt[i], x, srout*dt + W.yy[i+1] - W.yy[i], wout, target(Q)) # σ(t,x) (σ(t,x)' * tilder(t,x) + dW(t))
        Xᵒ.yy[i+1] .= x + dt * bout +  wout
        # likelihood terms
        if i<=length(tt)-1-skip
            _b!((i,tt[i]), x, btout, auxiliary(Q))
            som += dot(bout-btout, rout) * dt
            if !constdiff(Q)
                σt!(tt[i], x, rout, strout, auxiliary(Q))  #  tildeσ(t,x)' * tilder(t,x)
                som += 0.5*Bridge.inner(srout) * dt    # |σ(t,x)' * tilder(t,x)|^2
                som -= 0.5*Bridge.inner(strout) * dt   # |tildeσ(t,x)' * tilder(t,x)|^2
                Bridge.a!((i,tt[i]), x, A, target(Q))
                som += 0.5*(dot(At,Q.Ht[i]) - dot(A,Q.Ht[i])) * dt
            end
        end
    end
    som
end

if TEST
    Y = deepcopy(Xᵒ)
    Ynew = deepcopy(Xᵒ)
    @time Bridge.solve!(EulerMaruyama!(), Y, xinit, Wᵒ, Q)
    @time llikelihood(LeftRule(), Y, Q; skip = 1)


    @time simguidedlm_llikelihood!(LeftRule(), Ynew, xinit, Wᵒ, Q)
    j=30;print(Y.yy[j]-Ynew.yy[j])
end
