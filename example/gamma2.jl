#40*sum(Z.*phi.(40*(Z-0.04)))/x.tt[end]
using Bridge
using Base.Test
using Distributions
using Bridge: runmean

#srand(1234)
srand(123)

PYPLOT = false
PYPLOT && using PyPlot
import Bridge: increment, expint
import Distributions.pdf

N = 5 # number of thetas (so N+1 bins)
T = 2000.0
n = 10000 # number of increments
m = 20 # number of augmentation points per bridge exluding the left endpoint
beta0 = 0.4
beta0b = beta0/10
alpha0 = 2.0
alpha0b = alpha0/10

b10 = .5 # CPP values for testing
theta1 = -0.3

b1 = 0.1 # prior b1

if !isdefined(:simid)
    error("provide simid = {1,...,5}")
end

if simid == 1
    simname = "gammap" 
elseif simid == 2
    b10 = .5
    theta1 = -0.3
    simname = "gammappluscpp"
elseif simid == 3 
    simname = "sumgamma"
elseif simid == 4
    simname = "levymatters1"
    beta0 = 0.2
    alpha0 = 2.0
    N = 4
    T = 5000.0
    n = 5000 # number of increments
elseif simid == 5
    simname = "levymatters2"
    beta0 = 0.2
    alpha0 = 2.0
    N = 4
    T = 2500.0
    n = 50000 # number of increments
elseif simid == 6
    simname = "danishfire"
    beta0 = 1.8 # ml 
    alpha0 = 0.6
    dt = 0.02
    n = 500
    T = dt*n
    #alpha = 0.6
elseif simid == 7
    simname = "dfparam"
    beta0 = 1.8 # ml 
    alpha0 = 0.6
    dt = 0.02
    n = 500
    T = dt*n
    #alpha = 0.6
end

P0 = GammaProcess(beta0, alpha0)

tt = linspace(0, T, n + 1)
X0 = sample(tt, P0) 

P0b = GammaProcess(beta0b, alpha0b)
X0b = sample(tt, P0b) 



Z = diff(X0.yy)
dt = mean(diff(X0.tt))
PD = increment(dt, P0)

@test shape(PD) ≈ beta0*dt
@test scale(PD) ≈ 1/alpha0
@test beta0*dt/alpha0 ≈ mean(increment(dt, P0))
@test beta0*dt/alpha0^2 ≈ var(increment(dt, P0))

#=
m = mean(Z)
v = var(Z)

ahat = m/v
bhat = m^2/(dt*v)
=#

"""
    hist1(xx, r)

Count
"""
function hist1(xx, r; prob = false)
    c = zeros(Int, length(r) + 1)
    for x in xx
        c[first(searchsorted(r, x))] += 1
    end
    if prob
        c / sum(c)
    else
        c / 1
    end
end


pdf(x, beta, alpha) = beta/x*exp(-alpha*x)



# t(x) = x*(1+atan(x))
# Z2= t.(Z)
# X2 = SamplePath(X.tt, Bridge.cumsum0(Z2))

struct CPP
    b1
    γ
    alpha
    theta
    c
    CPP(b1, γ, alpha, theta) = theta > 0 ? throw(ArgumentError("positiv theta")):
        new(b1, γ, alpha, theta, γ * (expint(1, (alpha + theta) * b1) - expint(1, alpha * b1)))
end



# julia: density of Exp: 1/theta exp(-x/theta))
arrival(P::CPP) = Exponential(1/P.c)

function rjumpsize(P::CPP)
    c = P.c
    b1 = P.b1
    lambda = P.alpha + P.theta
    M = P.γ/(b1*c*lambda)*(exp(-lambda*b1))
    x = b1 + randexp()/lambda 
    while rand() <= jumppdf(x, P)/(M*lambda*exp(-lambda*(x-b1)))
        assert(0<= jumppdf(x, P)/(M*lambda*exp(-lambda*(x-b1))) <= 1.0)
        x = b1 + randexp()/lambda  
    end
    x
end

jumppdf(x, P) = x >= P.b1 ?  P.γ/(x*P.c)*(exp(-(P.alpha+P.theta)*x) - exp(-(P.alpha)*x)) : 0.0

function simulate(T, P::CPP)
    dt = rand(arrival(P))
    t = 0.0
    y = 0.0
    tt = [t]
    yy = [y]

    while t + dt <= T
        t = t + dt
        y = y + rjumpsize(P)
        append!(tt, t)
        append!(yy, y)

        dt = rand(arrival(P))
    end
    SamplePath(tt, yy)
end


CP = CPP(b10, beta0, alpha0, theta1)

xx = linspace(0, 40, 20000)
dx = xx[2]-xx[1]
@test abs(1 - sum(jumppdf.(xx, CP))*dx) < 0.002

@test abs(1/mean(rand(arrival(CP), 1000)) - CP.c) < 0.002

# test
#xx = linspace(0, 10, 200000)
dx = xx[2]-xx[1]
sum(xx.*jumppdf.(xx, CP))*dx, mean(rjumpsize(CP) for i in 1:100000)

Y = simulate(T, CP)


#plot(Y.tt, Y.yy)
# Plot jump process
if PYPLOT 
    plot(X0.tt, X0.yy)
    plot([0; repeat(Y.tt[2:end], inner=2); T], repeat(Y.yy, inner=2))
end

function addprc(X, Y)
    yy = copy(X.yy)
    for i in 1:length(yy)
        yy[i] += Y.yy[last(searchsorted(Y.tt, X.tt[i]))]        
    end
    SamplePath(X.tt, yy)
 end       

if simid == 2
    X = addprc(X0, Y)
elseif simid === 3 
    X = addprc(X0, X0b)
else
    X = X0
end


# grid points
# b = quantile(increment(dt, GammaProcess(beta0,alpha0)),(1:(N))/(N+1)) # theoretical
b = quantile(diff(X.yy), (N:2N-1)/(2N)) # first bin resembles 50% of emperical increment distributions.
#b = [Inf]
b = cumsum(0.3:0.4:2.0)
#println("P(Y < b1) = ", mean(diff(X.yy) .< b1))
h = hist1(diff(X.yy), b) # note that P(Y < b[1]) may be inaccurate if dt is chosen small
if any(h .< 20)
    warn("Less than 20 observations in bin")
end

if PYPLOT
    plot(XY.tt, XY.yy)
end

## Prior

vpi = 2.0
epi = 2.0

Pi = Gamma(epi^2/vpi, vpi/epi) # alpha
Pi2 = Normal(0., 1.0) # theta
Pi3 = Normal(0., 5.0) # rho

assert(mean(Pi) ≈ epi)
assert(var(Pi) ≈ vpi)



lpi(alpha, theta, rho) = logpdf(Pi, alpha) + sum(logpdf.(Pi2, theta)) + sum(logpdf.(Pi3, rho))

#initialize 

#...

# alpha, beta = truth


B = Vector{typeof(X)}(n)
#Bº = Vector{typeof(X)}(n)
tt = X.tt
yy = X.yy




# Bookkeeping
mkpath(joinpath("output", simname))
try # save cp of this file as documentation
    cp(@__FILE__(), joinpath("output",simname,"$simname.jl"); remove_destination=true)
end


beps = b[1]/5
#yy = diff(vcat(B...).yy)
#var(yy[yy.< b[1]/4])


iterations = 100_000
#beta0 = 0.8beta0
alpha = alpha0

if simid == 3
    beta = beta0 + beta0b
    alpha = (beta0*alpha0 + beta0b*alpha0b)/(beta0 + beta0b)
else
    beta = beta0
end

c = beta*(T/(n*m*alpha)) *(1-exp(-alpha*beps)) # compensator for small jumps


theta = zeros(N)
rho = zeros(N)

# prior chain step std
alphasigma = 0.15
thsigma = 0.05
rhosigma = 0.05*ones(N) # variance of rho1 = 0
rhosigma[1] = 0.0

open(joinpath("output", simname,"truth.txt"), "w") do f
    bn = join(["b$i" for i in 1:length(b)], " ")
    println(f, "N alpha0 beta0 T n $bn beps prior1 prior2 ") 
    println(f, N, " ", join(round.([alpha0, beta0, T, n, b... ],3)," "), " $beps \"$Pi\" \"$Pi2\" ") 
end

# initial augmentation

for i in 1:n
    Delta = tt[i+1]-tt[i]
    delta = Delta/m
    P0 = GammaProcess(beta, alpha)
    Pº = GammaBridge(tt[i+1], yy[i+1], P0)
    tti = linspace(tt[i], tt[i+1], m+1)
    B[i] = sample(tti, Pº, yy[i])
end
zz = copy(B[1].yy)
if PYPLOT
    clf()
    collect(plot(B[i].tt, B[i].yy) for i in 1:10)
    plot(X.tt[1:11], X.yy[1:11], "*")
end

# write header line for csv 
open(joinpath("output", simname, "params.txt"), "w") do f
    thn = join(["theta$i" for i in 1:length(theta)], " ")
    rhon = join(["rho$i" for i in 1:length(rho)], " ")  
    println(f, "n alpha $thn $rhon") 
end

P0 = GammaProcess(beta, alpha) 
P = LocalGammaProcess(P0, theta, rho, b)

mc = mcstart([alpha; theta; rho])
thacc = 0
Bacc = 0
alphaacc = 0
for iter in 1:iterations
    # logging
    mc = mcnext(mc, [alpha; theta; rho])
    open(joinpath("output", simname, "params.txt"), "a") do f
        println(f, iter, " ", join(round.([alpha; theta; rho], 8), " ")) 
    end
   
    # compensator for small jumps
    c = beta*(T/(n*m*alpha)) * (1 - exp(-alpha*beps))
    c = 0.0

    # sample bridges

    for i in 1:n
        Delta = tt[i+1]-tt[i]
        delta = Delta/m
        P0 = GammaProcess(beta, alpha)
        P = LocalGammaProcess(P0, theta, rho, b)
        Pº = GammaBridge(tt[i+1], yy[i+1], P0)
        #tti = linspace(tt[i], tt[i+1], m+1)
        Bº = SamplePath(B[i].tt, zz)
        sample!(Bº, Pº, yy[i])
        ll = llikelihood(Bº, P, c) - llikelihood(B[i], P, c) 
       # print(i, " ", ll, " ")
        if rand() < exp(ll)
            zz = B[i].yy
            B[i] = Bº
            Bacc += 1
        end
    end

    # sample parameters
    # update theta and rho
    if iter % 5 != 2 # remember to update formula for acceptane rates
        P0 = GammaProcess(beta, alpha)
        thetaº = theta + thsigma*randn(length(theta))
        rhoº = rho + rhosigma.*randn(length(rho))
        
        if thetaº[end] + alpha < eps() 
            # reject
        else
            P = LocalGammaProcess(P0, theta, rho, b)
            Pº = LocalGammaProcess(P0, thetaº, rhoº, b)

            ll = 0.0
            for i in 1:n
                ll += llikelihood(B[i], Pº, P, c)
            end
            print("$iter \t\t\t\t paramsº: ", round(ll, 5), " ", round.([thetaº; rhoº], 3))
            if rand() < exp(ll + lpi(alpha, thetaº, rhoº) - lpi(alpha, theta, rho))
                print("✓")
                theta = thetaº
                rho = rhoº
                thacc += 1
            end
            println()
        end
    end

    # todo: update alpha
    if  iter % 5 == 2 # remember to update formula for acceptane rates
        alphaº = alpha + alphasigma*randn()
        
        if alphaº < 0 || theta[end] + alphaº < eps() # according to Wilkinson
            # reject
        else   
            P0º = GammaProcess(beta, alphaº)
            P0 = GammaProcess(beta, alpha)
            
            Pº = LocalGammaProcess(P0º, theta, rho, b)
            P = LocalGammaProcess(P0, theta, rho, b)
            
            ll = 0.0
            for i in 1:n
                ll += llikelihood(B[i], Pº, P)
            end
            print("$iter \t alphaº: ", round(ll, 5), " [", round(alphaº, 3), "]")
            if rand() < exp(ll + lpi(alphaº, theta, rho) - lpi(alpha, theta, rho))
                alpha = alphaº
                alphaacc += 1
                print("✓")
            end
            println() 
        end  
    end

end    


println("alpha acc ", alphaacc/(iterations/5))
println("theta acc ", thacc/(4iterations/5))
println("posterior band")
display(hcat(mcband(mc)...))
println("posterior band for mean")
display(hcat(mcbandmean(mc)...))

params = readdlm(joinpath("output", simname, "params.txt"), Float64; skipstart=1)[:,2:end];

function plotparams(simname = "gammap") 
    params = readdlm(joinpath("output", simname, "params.txt"), Float64; skipstart=1)[:,2:end];
    clf();plot(params[:,1], color=:blue , lw = 0.5)

    plot(params[:,2:end], color=:lightblue , lw = 0.2)
    plot(runmean(params), color=:grey, lw = 0.2)

    params
end
