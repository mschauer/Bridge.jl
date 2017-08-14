import Base.eltype
@deprecate statetype(P::ContinuousTimeProcess) valtype(P) 
@deprecate eltype(X::SamplePath) valtype(X) 
@deprecate setv(X::SamplePath, v) endpoint!(X, v)


bridge(W::SamplePath, P, scheme! = euler!) = bridge!(copy(W), W, P, scheme!)
function bridge!(Y::SamplePath, W::SamplePath, P, scheme! = euler!)
    !(W.tt[1] == P.t0 && W.tt[end] == P.t1) && error("Time axis mismatch between bridge P and driving W.") # not strictly an error
    scheme!(Y, P.v0, W, P)
    Y.yy[.., length(W.tt)] = P.v1
    Y
end