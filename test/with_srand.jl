# run last
using Bridge, Base.Test, StaticArrays

srand(1)
tt = collect(linspace(0.0, 1.0))
W = sample(tt, Wiener{SVector{3,Float64}}())
VW = VSamplePath(tt, copy(Bridge.mat(W.yy)))
srand(1)
sample!(VW, Wiener())


@test maximum(abs.(VW.yy - Bridge.mat(W.yy))) < eps()
