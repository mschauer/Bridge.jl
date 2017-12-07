using PyPlot, Bridge
λ(t) = 6/(t + 2)
T = 10.0
λmax = 3.0

P = InhomogPoisson(λ)
tt = sample(ThinningAlg(λmax), T, P)

plot(Bridge.piecewise(SamplePath(tt, collect(1:length(tt))))..., label="N(t)")
legend()

