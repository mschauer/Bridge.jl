
using Test
out1 = copy(xT)
@time out1 = (Bridge.b!(t[end], xT, out1, Pmsaux))

@time B = Bridge.B(t[end], Pmsaux)
x = vec(xT)
out2 = copy(x)
@time out2 = mul!(out2, B, x, true, false)

@test vec(out1) ≈ out2

Bridge.a(1.1,Pmsaux)
