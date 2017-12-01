
clf()
plot(X0a.tt[1:10:div(end,10)]*10, X0a.yy[1:10:div(end,10)]*10, label="c")
plot(X0b.tt[1:100:end], X0b.yy[1:100:end], label="b")
plot(X0a.tt[1:1:end], X0a.yy[1:1:end], label="a")
legend()

