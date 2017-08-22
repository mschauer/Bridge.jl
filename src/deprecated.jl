import Base.eltype
@deprecate statetype(P::ContinuousTimeProcess) valtype(P) 
@deprecate eltype(X::SamplePath) valtype(X) 
@deprecate setv(X::SamplePath, v) endpoint!(X, v)


@deprecate euler(u, W, P) solve!(Euler(), copy(W), u, W, P)

"""
    euler(u, W, P) -> X
  
Solve stochastic differential equation ``dX_t = b(t,X_t)dt + σ(t,X_t)dW_t`` using the Euler scheme.
"""
euler

@deprecate euler!(Y, u, W::SamplePath, P::ContinuousTimeProcess{T}) where {T}  solve!(Euler(), Y, u, W, P)


"""
    euler!(Y, u, W, P) -> X
  
Solve stochastic differential equation ``dX_t = b(t,X_t)dt + σ(t,X_t)dW_t`` 
using the Euler scheme in place.
"""
euler!


@deprecate innovations(Y, P)  innovations!(Euler(), copy(Y), Y, P)
@deprecate innovations!(W, Y::SamplePath, P) innovations!(Euler(), W, Y, P)



@deprecate mdbinnovations(Y, P)  innovations!(Mdb(), copy(Y), Y, P)
@deprecate mdbinnovations!(W, Y::SamplePath, P) innovations!(Mdb(), W, Y, P)

@deprecate bridge(W::SamplePath, P, scheme! = euler!) bridgeold!(copy(W), W, P, scheme!)
@deprecate bridge!(Y::SamplePath, W::SamplePath, P, scheme! = euler!) bridgeold!(Y, W, P, scheme!)

function bridgeold!(Y::SamplePath, W::SamplePath, P, scheme! = euler!)
    !(W.tt[1] == P.t0 && W.tt[end] == P.t1) && error("Time axis mismatch between bridge P and driving W.") # not strictly an error
    scheme!(Y, P.v0, W, P)
    Y.yy[.., length(W.tt)] = P.v1
    Y
end

