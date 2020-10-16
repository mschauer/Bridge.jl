
using Bridge, ConjugatePriors, Plots, Distributions, LaTeXStrings

srand(5);
pgfplots()

simname = :gammachain
simid = 1

#joinpath(ENV["BRIDGE_OUTDIR"],  "$simname$(simid)paths.jld2")


Nf = 800001 # divisible by 25, 50, 100
step = [25, 50, 100, 200][4] # subsampling stepsize
H = [20,40,80,160,320][4]

skipp = 100

s1(t) = 3/2 + sin(2*(4t-2)) + 2exp(-16((4t-2)).^2) # called s_1 in the paper
b(x) = -10x + 20. # called b_1 in the paper


ss = [0.1,0.13,0.15,0.23,0.25,0.4,0.44,0.65,0.76,0.78,0.81]
h = [4,-5,3,-4,5,-4.2,2.1,4.3,-3.1,2.1,-4.2]
s2(t) = sum(h[])


s = s1
dtf = 1/(Nf-1)
ttf = 0:dtf:1

N = (Nf-1) ÷ step + 1 # number of observations: n
td = (0:dtf:1)[1:step:end]    
dt = td[2] - td[1]

M = (length(td)-1) ÷ H # observed intervals per bin
pri = InverseGamma(0.1, 0.1)

r = length(td)-1 - H*M
assert(H*M == length(td)-1)

a1 = 1/1000
a, aζ = 1000.0/sqrt(H), 1000.0/sqrt(H)


import Bridge.piecewise

function piecewise(Y::SamplePath, tend = Y.tt[end])
    tt = [Y.tt[1]]
    n = length(Y.yy)
    append!(tt, repeat(Y.tt[2:n], inner=2))
    push!(tt, tend)
    tt, repeat(Y.yy, inner=2)
end
function piecewise(X::Bridge.EstSamplePath, tend = X.tt[end])
    Bridge.EstSamplePath(Bridge.piecewise(SamplePath(X.tt, X.yy), tend)..., 
        Bridge.piecewise(SamplePath(X.tt, X.vv), tend)[2])
end

@recipe function plot(X::Bridge.EstSamplePath{Float64}; quantile = 0.95)
    q = Distributions.quantile(Normal(), 1 - (1 - quantile)/2)
    if Plots.backend() != Plots.PGFPlotsBackend()
        ribbon --> q*sqrt.(X.vv)
        X.tt, X.yy
    else
        ribbon --> (NaN,NaN)
        linewidth --> 0
        vcat(X.tt, reverse(X.tt)), vcat(X.yy + q*sqrt.(X.vv), reverse(X.yy - q*sqrt.(X.vv)))
    end
end

@recipe function plot(X::SamplePathBand)
    if Plots.backend() != Plots.PGFPlotsBackend()
        error("todo")
    else
        ribbon --> (NaN,NaN)
        linewidth --> 0
        vcat(X.tt, reverse(X.tt)), vcat(X.uu, reverse(X.ll))
    end
end



"""
White noise on the grid tt
"""
dW(tt) = randn(length(tt)-1).*sqrt.(diff(tt))

"""
Finite difference operator 
"""
function diff0(x) 
    y = similar(x)
    y[1] = zero(x)
    for i in 2:length(x)
        y[i] = x[i]-x[i-1]
    end
    y
end

""" 
Euler scheme with start in y1, space dependent drift b and time dependent dispersion coefficient
"""
function euler1(tt, y1, b, sigma, yy = zeros(tt), dW = dW(tt))
    yy[1] = y1
    for i in 2:length(yy)
        yy[i] = yy[i-1] + b(yy[i-1]) * (tt[i]-tt[i-1]) + sigma(tt[i-1])*dW[i-1]
    end
    yy
end



   
Xf = euler1(ttf, 0., b, s, zeros(ttf), dW(ttf))
;

Xd = Xf[1:step:end]
   
                

ttplot = ttf[1:skipp:end]
Xplot = Xf[1:skipp:end]
plot(ttplot, Xplot, label=L"X_t")



post = Vector(H)
Z = Vector(H)


for i in 1:H
    ii = 1+(i-1)*M:(i)*M+1
    tx = td[ii]
    x = diff(Xd[ii])./sqrt.(diff(tx)) # normalize obs
    Z[i] = sum(diff(Xd[ii]).^2)
    post[i] = ConjugatePriors.posterior((0., pri), Normal, x)
end

@show rate(post[end])
@show rate(InverseGamma(0.1 + M/2, 0.1 + Z[end]/2/dt))

xhat = sqrt.(median.(post));


plot(Bridge.piecewise(SamplePath(td[1:M:end],xhat)))
plot!(ttplot, s.(ttplot))



v = zeros(H)
v[:] = xhat
#ζ = zeros(H + 1);
ζ = zeros(H);

v[1] = rand(InverseGamma(a1, a1))

for i in 2:H
    ζ[i] = rand(InverseGamma(aζ, inv(v[i-1]/aζ)))
    v[i] = rand(InverseGamma(a, inv(ζ[i]/a)))
end

plot(v[1:end])



iterations = 10000
state = Bridge.mcstart(v)
for iter in 1:iterations
    # sample z
    ζ[1] = rand(InverseGamma(aζ + a + 1, aζ/bζ + a/v[1]))
    for h in 2:H
        ζ[h] = rand(InverseGamma(aζ + a, (aζ/v[h-1] + a/v[h])))
    end   
  #  ζ[1] = 1/rand(Gamma(aζ + a, aζ/v[H] + a/v[1]))
    for h in 1:H-1
        v[h] = rand(InverseGamma(aζ + a + M/2, (a/ζ[h] + aζ/ζ[h+1] + Z[h]/2/dt)))
    end  
    v[H] = rand(InverseGamma(aζ + (M+r)/2, a/ζ[H] + Z[H]/2/dt))
    state = Bridge.mcnext(state, v)
end
v;



plot(Bridge.EstSamplePath(td[1:M:end-1], Bridge.mcstats(state)...), size = (900,300),
    yaxis =((0,20),0:5:20), color=colorant"#0044FF", label = "marginal 95\\% cov.~band", layout=2, subplot=1)
plot!(ttplot, (s.(ttplot)).^2, linewidth=2, color="red", label=L"s_2", subplot=1)
plot!(Bridge.EstSamplePath(td[1:M:end-1], mean.(post), var.(post) ), 
    yaxis = ((0,20),0:5:20), color=colorant"#0044FF", label = "marginal 95\\% cov.~band", layout=2, subplot=2)
plot!(ttplot, (s.(ttplot)).^2, linewidth=2, color="red", label = L"s_2", subplot=2)

#savefig("chain.pdf")


plot(piecewise( Bridge.EstSamplePath(td[1:M:end], Bridge.mcstats(state)...) ), size = (900,300),
    yaxis =((0,20),0:5:20), color=colorant"#0044FF", label = "marginal 95\\% cov.~band", layout=2, subplot=1)
plot!(ttplot, (s.(ttplot)).^2, linewidth=2, color="red", label=L"s_2", subplot=1)
plot!(piecewise( Bridge.EstSamplePath(td[1:M:end], mean.(post), var.(post)) ), 
    yaxis = ((0,20),0:5:20), color=colorant"#0044FF", label = "marginal 95\\% cov.~band", layout=2, subplot=2)
plot!(ttplot, (s.(ttplot)).^2, linewidth=2, color="red", label = L"s_2", subplot=2)





diff(td[1:M:M+1]),
1/H

#length(td[1:M:end-1]), length(Bridge.mcstats(state)[1])

#plot(piecewise(Bridge.EstSamplePath(td[1:M:end], Bridge.mcstats(state)...)))


