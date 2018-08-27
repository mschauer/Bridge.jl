using Bridge, Distributions, StaticArrays

PLOT = :winston
include("plot.jl")

tt = range(0, stop=1, length=1000)
G =  GammaProcess(10., 1.)
X = sample(tt, G)
plot(X, "-")
GB =  GammaBridge(1., 1., G)
XB = sample(tt, GB)
oplot(XB, "-")

VG =  VarianceGammaProcess(0., 1., 1.)
X = sample(tt, VG)
plot(X, "-")

#n = 1000
#simulate gamma process:
#tt = cumsum(rand(Gamma(0.01, 1/(0.01n)), n))
#plot(tt)


X = sample(range(0, stop=1, length=500),Bridge.GammaBridge(1,1, GammaProcess(10,10)))
X = sample(range(0, stop=1, length=500), GammaProcess(10,10))
