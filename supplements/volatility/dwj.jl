srand(5)
include("setup.jl")
simid = 2
simname = "dwj"
hyper = false
data = readcsv("output/volatility/dwj.csv", header=true)[1]
Xd = Vector{Float64}(log.(data[:, 2]/data[1, 2]))
quc = 90
qu = quc/100

td = map(d -> Dates.value(Dates.Date(d)), data[:,1])/1
dates = map(d -> Dates.value(Dates.Date(d)), data[:,1])/365
dates += -dates[1] + 1971 + (Dates.value(Date("1971-07-02")) - Dates.value(Date("1971-01-01")))/365

td -= td[1]
dt = td[2] - td[1]

#assert(all(diff(td) .== 7))
X = data[:,2][:]
sub = 10
lim = (740, 1060)
display(plot(dates, X,  xaxis = ("y", (dates[1],dates[end]), 1972:1974), linewidth = 0.4, ylim = lim, size = (400, 400/1.4), color=colorant"#0044FF",  label=L"X"))
savefig("output/volatility/fig$(simname)pathX.pdf")
lim = (-0.2, 0.2)
display(plot(dates, Xd,  xaxis = ("y", (dates[1],dates[end]), 1972:1974), linewidth = 0.4, ylim = lim, size = (400, 400/1.4), color=colorant"#0044FF",  label=L"Z"))
savefig("output/volatility/fig$(simname)pathZ.pdf")
lim = (-0.1, 0.1)
display(plot(dates[1:end-1], diff(X)./X[1:end-1], xaxis = ("y", (dates[1],dates[end]), 1972:1974), linewidth = 0.4, ylim = lim, size = (400, 400/1.4), color=colorant"#0044FF",  label=L"Y"))
savefig("output/volatility/fig$(simname)pathY.pdf")



N = length(Xd)

if simid == 1
    a1 = 0.0
    a = 5.0
    aζ = a

    M = 6
    hyper = true

    σa = 10.0
elseif simid == 2
    a1 = 0.0
    a = 5.0
    aζ = a

    M = 12
    hyper = true

    σa = 10.0
elseif simid == 3
    a1 = 0.0
    a = 5.0
    aζ = a

    M = 6
    hyper = true

    σa = 12.0
    Xd = X
elseif simid == 4
    a1 = 0.0
    a = 5.0
    aζ = a

    M = 12
    hyper = true

    σa = 12.0   
    Xd = X
else 
    error("")
end

pri = InverseGamma(0.1, 0.1)
priind= InverseGamma(0.001,0.001)
pri2 = InverseGamma(0.3, 0.3) 
#pri2 = InverseGamma(2.5, 2.5) #try

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

# compute independent posterior

post = Vector(H)
Z = Vector(H)
x = Vector(H)
for i in 1:H
    if i == H
        ii = 1+(i-1)*M:N
    else
        ii = 1+(i-1)*M:(i)*M+1
    end
    tx = td[ii]

    x[i] = diff(Xd[ii])./sqrt.(diff(tx)) # normalize obs
    Z[i] = sum(diff(Xd[ii]).^2)
    post[i] = ConjugatePriors.posterior((0., priind), Normal, x[i])
 #   post[i] = InverseGamma(shape(priind) + (M + (i==H)*r)/2, scale(priind) + Z[i]/2/dt)
  
end



if simid <= 2
    lim = (-0.01, .042)
else 
    lim = (-2, 22)
end
bands = map(p->sqrt(quantile(p, (1 - qu)/2)), post), map(p->sqrt(quantile(p, 1 - (1 - qu)/2)), post)

plot(piecewise( Bridge.SamplePathBand(dates[1:M:end], bands...) ), 
    ylim = lim, xaxis = ("y", (dates[1],dates[end]), 1972:1974), size = (400, 400/1.4), color=colorant"#0044FF", label = "marginal $quc\\% cred.~band",); 
display(Plots.current())
savefig("output/volatility/fig$(simname)postind$simid.pdf")


v = zeros(H)
ζ = zeros(H)

# initialize MCMC with a draw from the prior
if a1 > 0.01 
    v[1] = rand(InverseGamma(a1, a1))
else 
    v[1] = 1.0
end
for i in 2:H
    ζ[i] = rand(InverseGamma(aζ, inv(v[i-1]/aζ)))
    v[i] = rand(InverseGamma(a, inv(ζ[i]/a)))
end

subsampl = 1
burnin = 1000
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
        mod(iter, printiter) == 0 && print("$iter \t α ", a˚, (lq˚,lq))
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
if simid <= 2
    lim = (-0.01, .027)
else 
    lim = (-2, 22)
end
quc = 90
qu = quc/100
bands = mapslices(v->sqrt(quantile(v, (1-qu)/2)), samples, (2,))[:],  mapslices(v->sqrt(quantile(v, 1 - (1 - qu)/2)), samples, (2,))[:]


plot(piecewise( Bridge.SamplePathBand(dates[1:M:end], bands...) ), 
    ylim = lim, xaxis = ("y", (dates[1],dates[end]), 1972:1974), size = (400, 400/1.4),color=colorant"#0044FF", label = "marginal $quc\\% cred.~band",); 
display(Plots.current())
savefig("output/volatility/fig$(simname)post$simid.pdf")



open(joinpath("output", "volatility", "$(simname)$(simid)info.txt"), "w") do f
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

