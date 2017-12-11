using PyPlot
figure(figsize=(8,5))
fill_between(X.tt,  first.(Xmeanm + 1.96*Xstdm), first.(Xmeanm - 1.96*Xstdm), edgecolor=:darkblue, facecolor=:lightblue, hatch="X", label="95% cred. band")

plot(X.tt, first.(X.yy), label=L"X_1", color=:darkviolet)


plot(X.tt, first.(Xmeanm), label=L"x_1", color=:darkgreen)


if partial
    plot(V.tt, V.yy, "*", label=L"V",  color=:darkviolet)
else
    plot(V.tt, first.(V.yy), ",", label=L"V_1",  color=:darkviolet)
end
PyPlot.legend()

figure(figsize=(8,5))


for i in 1:length(Paths)
    plot(X.tt[1:end-1], first.(Paths[i]), color=:lightblue)
end
plot(X.tt, first.(X.yy), label=L"X_1", color=:darkviolet)
plot(X.tt, first.(Xmeanm), label=L"x_1", color=:darkgreen)

if partial
    plot(V.tt, V.yy, "*", label=L"V",  color=:darkviolet)
else
    plot(V.tt, first.(V.yy), ",", label=L"V_1",  color=:darkviolet)
end
PyPlot.legend()


figure(figsize=(5,5))

for i in 1:length(Paths)
    plot(first.(Paths[i]), last.(Paths[i]), color=:lightblue)
end
plot(first.(X.yy), last.(X.yy), label=L"X", color=:darkviolet)

plot(first.(Xmeanm), last.(Xmeanm), label=L"x_1", color=:darkgreen)

PyPlot.legend()


figure(figsize=(8,5))
plot(X.tt, last.(X.yy), label=L"X_2", linewidth=0.2, color=:darkviolet)
plot(X.tt, last.(Xmeanm), ":", label=L"x_2", color=:darkgreen)
fill_between(X.tt,  last.(Xmeanm + 1.96*Xstdm), last.(Xmeanm - 1.96*Xstdm), edgecolor=:darkblue, facecolor=:lightblue, hatch="X", label="95% cred. band")

if !partial
    plot(V.tt, last.(V.yy), "*", label=L"V_2",  color=:darkviolet)
end
PyPlot.legend()
