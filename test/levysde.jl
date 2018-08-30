# Simulate an Levy driven SDE

using Bridge, PyPlot
Random.seed!(5.)

# Define Gamma process
γ = 10.
α = 5.
P = GammaProcess(γ, α)

# Define Grid

T = 5.0
tt = 0.0:T/n:T
m = 1000
sigma = (t, x = 0.0) -> exp(-cos(t))

# Driving Levy process is a difference of Gamma processes
L = sample(tt, P) .- sample(tt, P)

# Solve Levy driven SDE
X = solve(EulerMaruyama(), 0.0, L, ((t,x)-> -0.*x, sigma))


# Plot result
figure();plot(X.tt, X.yy, label="X")
plot(X.tt, L.yy, label="L")
plot(X.tt, sigma.(X.tt), label=L"\sigma")
legend()
PyPlot.savefig("output/levysde.png")
