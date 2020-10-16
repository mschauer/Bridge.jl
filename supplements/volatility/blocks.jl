include("setup.jl")

srand(5)
simname = "blocks"
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

    σa = 1.0
elseif simid == 4
    a1 = 1/10
    a = 20.0
    aζ = a

    M = 12
    hyper = true

    σa = 1.5 
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

ss = [0.1,0.13,0.15,0.23,0.25,0.4,0.44,0.65,0.76,0.78,0.81]
const hh = 3.655606*[4,-5,3,-4,5,-4.2,2.1,4.3,-3.1,2.1,-4.2]

tx = linspace(0, 1, 1000)
s3(t) = 10 + sum(hh[1:last(searchsorted(ss, t))])

display(plot(tx, s3.(tx), ylim = (0, 31), size = (400, 400/1.4), color=colorant"#0044FF",  label = LaTeXString("\$s_3\$")))

savefig("output/volatility/fig$simname.pdf")

Xf = euler1(ttf, 0., b, s3, zeros(ttf), dW(ttf))
Xd = Xf[1:step:end]
   
sub = 10
lim = (-5.2, 15)
display(plot(td[1:sub:end], Xd[1:sub:end], linewidth = 0.4, ylim = lim, size = (400, 400/1.4), color=colorant"#0044FF",  label=L"X"))
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



lim = (-2, 80)
bands = map(p->sqrt(quantile(p, (1 - qu)/2)), post), map(p->sqrt(quantile(p, 1 - (1 - qu)/2)), post)

ttplot = linspace(0, 1, 1000)[2:end-1]

plot(ttplot, s3.(ttplot), ylim = lim, size = (400, 400/1.4), linewidth = 0.7, color="red", label=L"s");
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

lim = (-2, 80)

bands = mapslices(v->sqrt(quantile(v, (1-qu)/2)), samples, (2,))[:],  mapslices(v->sqrt(quantile(v, 1 - (1 - qu)/2)), samples, (2,))[:]
ttplot = linspace(0, 1, 1000)[2:end-1]

plot(ttplot, s3.(ttplot), ylim = lim, size = (400, 400/1.4), linewidth = 0.7, color="red", label=L"s");
plot!(piecewise( Bridge.SamplePathBand(td[1:M:end], bands...) ), 
    color=colorant"#0044FF", label = "marginal $quc\\% cred.~band",); 
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

