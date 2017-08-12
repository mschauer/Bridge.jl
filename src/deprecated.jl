import Base.eltype
@deprecate statetype(P::ContinuousTimeProcess) valtype(P) 
@deprecate eltype(X::SamplePath) valtype(X) 
@deprecate setv(X::SamplePath, v) endpoint!(X, v)