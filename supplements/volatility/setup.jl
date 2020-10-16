
using Bridge, ConjugatePriors, Plots, Distributions, LaTeXStrings

srand(5);
pgfplots()

simname = :gammachain

#joinpath(ENV["BRIDGE_OUTDIR"],  "$simname$(simid)paths.jld2")


Nf = 800001 # divisible by 25, 50, 100
step = [25, 50, 100, 200][4] # subsampling stepsize
H = [20,40,80,160,320][4]

skipp = 100

s1(t) = 3/2 + sin(2*(4t-2)) + 2exp(-16((4t-2)).^2) # called s_1 in the paper
b(x) = -10x + 20. # called b_1 in the paper


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

a1 = 1/10
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

@recipe function plot(X::Bridge.SamplePathBand)
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

