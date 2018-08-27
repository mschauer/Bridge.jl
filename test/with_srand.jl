# run last
using Bridge, Test, StaticArrays

Random.seed!(1)
tt = collect(range(0.0, stop=1.0, length=50))
W = sample(tt, Wiener{SVector{3,Float64}}())
VW = VSamplePath(tt, copy(Bridge.mat(W.yy)))
Random.seed!(1)
sample!(VW, Wiener())


@test maximum(abs.(VW.yy - Bridge.mat(W.yy))) < eps()
