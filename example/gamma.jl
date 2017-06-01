using Bridge, Distributions, StaticArrays

PLOT = :winston
include("plot.jl")

tt = linspace(0,1, 1000)
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


