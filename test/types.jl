using Test
using Bridge
X = SamplePath(0.0:1.0:10.0,Bridge.cumsum0(rand(10)))

#@test_throws SamplePath(0.0:1.0:10.0,Bridge.cumsum0(rand(10)))

@test length(X) == length(X.tt)

@test collect(t for (t,x) in X) == X.tt
@test collect(x for (t,x) in X) == X.yy

@test Base.iteratorsize(SamplePath) == Base.HasLength()
