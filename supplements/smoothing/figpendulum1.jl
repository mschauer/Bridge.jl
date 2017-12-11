using PyPlot
partial = true
include("pendulum.jl")

clf()
plot(X.tt, first.(X.yy), label=L"X_1", color=:darkviolet)
plot(X.tt, last.(X.yy), label=L"X_2", linewidth=0.2, color=:darkviolet)


plot(X.tt, first.(Xsm.yy), label=L"x_1", color=:darkgreen)
plot(X.tt, last.(Xsm.yy), ":", label=L"x_2", color=:darkgreen)

if partial
    plot(V.tt, V.yy, "*", label=L"V",  color=:darkviolet)
else
    plot(V.tt, first.(V.yy), ",", label=L"V_1",  color=:darkviolet)
    plot(V.tt, last.(V.yy), "*", label=L"V_2",  color=:darkviolet)
end
legend()


