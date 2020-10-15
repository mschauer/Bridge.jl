
using Bridge, ConjugatePriors, Plots, Distributions, LaTeXStrings
import Bridge: b, σ

srand(5);
pgfplots()

struct CIR <: ContinuousTimeProcess{Float64}
    η::Tuple{Float64,Float64,Float64}
end
b(s, x, P::CIR) = P.η[1] + P.η[2]*x
σ(s, x, P::CIR) = P.η[3]*sqrt(max(x, 0.0))

P = CIR((6.0, -3.0, 2.0))

x0 = 1.0


Nf = 800001 # divisible by 25, 50, 100
step = [25, 50, 100, 200][4] # subsampling stepsize
H = [20,40,80,160,320][4]

skipp = 100

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



srand(4)
simname = "CIR"
hyper = false

quc = 95
qu = quc/100

if simid == 1
    a1 = 1/10
    a = 20.0
    aζ = a

    M = 25
elseif simid == 2
    a1 = 1/10
    a = 40.0
    aζ = a

    M = 12
elseif simid == 5
    a1 = 1/10
    a = 40.0
    aζ = a

    M = 25
elseif simid == 6
    a1 = 1/10
    a = 20.0
    aζ = a

    M = 12   

elseif simid == 3
    a1 = 1/10
    a = 20.0 # initialisation
    aζ = a

    M = 25
    hyper = true

    σa = 10.0
elseif simid == 4
    a1 = 1/10
    a = 20.0
    aζ = a

    M = 12
    hyper = true

    σa = 15.0 
else 
    error("")
end

pri = InverseGamma(0.1, 0.1)
pri2 = InverseGamma(0.3, 0.3)

if hyper
    iterations = 200000
else
    iterations = 200000
end
#
#M = (length(td)-1) ÷ H # observed intervals per bin
#r = length(td)-1 - H*M
#assert(H*M == length(td)-1)
H = (N-1)÷M
r = length(td)-1 - H*M


tx = linspace(0, 1, 1000)


W = sample(ttf, Wiener())


Xf = solve(Euler(), 1., W, P).yy
Xd = Xf[1:step:end]
Xfplot = Xf[1:500:end]

ss = σ.(ttf, Xf, P)
ttplot = ttf[1:500:end]
ssplot = ss[1:500:end]


display(plot(ttplot, ssplot, ylim = (0.0, 6.2), linewidth = 0.6, size = (400, 400/1.4), color=colorant"#0044FF",  label = LaTeXString("\$s(\\omega)\$")))

savefig("output/volatility/fig$simname.pdf")
   
sub = 10
lim = (0.0, 6.2)
display(plot(ttplot, Xfplot, linewidth = 0.6, ylim = lim, size = (400, 400/1.4), color=colorant"#0044FF",  label=L"X"))
savefig("output/volatility/fig$(simname)path.pdf")

# compute independent posterior

post = Vector(H)
Z = Vector(H)

for i in 1:H
    if i == H
        ii = 1+(i-1)*M:N
    else
        ii = 1+(i-1)*M:(i)*M+1
    end
    tx = td[ii]

    x = diff(Xd[ii])./sqrt.(diff(tx)) # normalize obs
    Z[i] = sum(diff(Xd[ii]).^2)
    post[i] = ConjugatePriors.posterior((0., pri), Normal, x)
end



lim = (-0.2, 8.2)
bands = map(p->sqrt(quantile(p, (1 - qu)/2)), post), map(p->sqrt(quantile(p, 1 - (1 - qu)/2)), post)


plot(ttplot, ssplot, ylim = lim, size = (400, 400/1.4), linewidth = 0.7, color="red", label=L"s");
plot!(piecewise( Bridge.SamplePathBand(td[1:M:end], bands...) ), 
    color=colorant"#0044FF", label = "marginal $quc\\% cred.~band",); 
display(Plots.current())
savefig("output/volatility/fig$(simname)postind$simid.pdf")


v = zeros(H)
ζ = zeros(H)

# initialize MCMC with a draw from the prior

v[1] = rand(InverseGamma(a1, a1))
for i in 2:H
    ζ[i] = rand(InverseGamma(aζ, inv(v[i-1]/aζ)))
    v[i] = rand(InverseGamma(a, inv(ζ[i]/a)))
end

subsampl = 1
burnin = 0
subinds = (1 + burnin):subsampl:iterations
samples = zeros(H, length(subinds))

state = Bridge.mcstart(v)
si = 1
printiter = 10
as = []
acc = 0
for iter in 1:iterations
    # move: parameters
    v[1] = rand(InverseGamma(a1 + aζ + M/2, a1 + aζ/ζ[2] + Z[1]/2/dt))
    for h in 2:H-1
        v[h] = rand(InverseGamma(aζ + a + M/2, (a/ζ[h] + aζ/ζ[h+1] + Z[h]/2/dt)))
    end  
    v[H] = rand(InverseGamma(aζ + (M+r)/2, a/ζ[H] + Z[H]/2/dt))

    # move: latent
    if iter in subinds
        samples[:, si] = v
        si += 1
    end
    for h in 2:H
        ζ[h] = rand(InverseGamma(aζ + a, (aζ/v[h-1] + a/v[h])))
    end 

    state = Bridge.mcnext(state, v)

    # move: hyper parameters
    if hyper
        a˚ = a + σa*randn() 
        while a˚ < eps()
            a˚ = a + σa*randn() 
        end     

        lq = logpdf(pri2, a)
        lq += (2*(H-1))*(a*log(a) - lgamma(a))
        s = sum(log(v[k-1]*v[k]*ζ[k]*ζ[k]) + (1/v[k-1] + 1/v[k])/ζ[k]  for k = 2:H)
        lq += -a*s

        lq˚ = logpdf(pri2, a˚)
        lq˚ += (2*(H-1))*(a˚*log(a˚) - lgamma(a˚))
        lq˚ += -a˚*s
        mod(iter, printiter) == 0 && print("$iter \t α ", a˚)
        if rand() < exp(lq˚ - lq)*cdf(Normal(0, σa), a)/cdf(Normal(0, σa), a˚)
            acc = acc + 1
            aζ = a = a˚
            mod(iter, printiter) == 0 && print("✓")
        end
        mod(iter, printiter) == 0 && println()
        push!(as, a)
    end

end
hyper && println("acc prob hyper ", acc/iterations)
writecsv("output/volatility/samples$simname$simid.csv", samples')
if hyper
    writecsv("output/volatility/samplesas$simname$simid.csv", as)
end    

lim = (-0.2, 6.2)

bands = mapslices(v->sqrt(quantile(v, (1-qu)/2)), samples, (2,))[:],  mapslices(v->sqrt(quantile(v, 1 - (1 - qu)/2)), samples, (2,))[:]

plot(ttplot, ssplot, ylim = lim, size = (400, 400/1.4), linewidth = 0.7, color="red", label=L"s");
plot!(piecewise( Bridge.SamplePathBand(td[1:M:end], bands...) ), 
    color=colorant"#0044FF", label = "marginal $quc\\% cred.~band",); 
display(Plots.current())
savefig("output/volatility/fig$(simname)post$simid.pdf")



open(joinpath("output", "volatility", "$(simname)$(simid)info.txt"), "w") do f
    println(f, "η ", P.η)
    println(f, "α1 ", a1)
    !hyper && println(f, "α ", a)
    println(f, "M ", M)
    println(f, "iterations ", iterations)
    println(f, "samples at ", subinds)
    println(f, "$quc % credible bands")
    
    println(f, "hyper ", hyper)
    if hyper
        println(f, "acc ", round(acc/iterations, 3)) 
        println(f,  "σa ", σa)
    end
end


# TODO
# use quantiles
# add samplepath 

