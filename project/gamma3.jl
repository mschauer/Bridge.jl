#40*sum(Z.*phi.(40*(Z-0.04)))/x.tt[end]
using Bridge
using Base.Test
using Distributions
using Bridge: runmean, lp

#srand(1234)
srand(123)

PYPLOT = false
PYPLOT && using PyPlot
import Bridge: increment, expint, lp
import Distributions.pdf

N = 5 # number of thetas (so N+1 bins)
#b = cumsum(0.3:0.4:2.0)
if !isdefined(:simid)
    error("provide simid = {1,2,3}")
end

sim = [:gamma, :sumgamma, :fire][simid]

if sim == :fire
    #b = [0.4, 1.2, 3.6]
    b = [0.75, 2.5, 3.6]
    b = [0.5, 1.5, 3]
    b = [0.5, 1, 1.5, 2, 2.5, 3., 3.5]
    b = [0.75, 2, 4]
    b = [0.75, 2.5, 3.6]
    b = [1.0, 2.0, 4.0]
    #b = [1.0, 1.5, 3.5, 5]
#    b = [3.0, 5.0]
    N = length(b)
end

T = 2000.0
n = 10000 # number of increments
m = 20 # number of augmentation points per bridge exluding the left endpoint
if sim == :fire
    m = 400
end

tt = linspace(0.0, T, n + 1)


iterations = 200_000


# two gamma processes
beta0a = 0.4
beta0b = beta0a/10
alpha0a = 2.0
alpha0b = alpha0a/10



transdim = true
fixalpha = transdim

if sim == :gamma
    simname = "gammap"
    alpha0 = alpha0a
    beta0 = beta0a
elseif sim == :sumgamma
    simname = "sumgamma"
    beta0 = beta0a + beta0b
    alpha0 = (beta0a*alpha0a + beta0b*alpha0b)/(beta0a + beta0b)
    
elseif sim == :fire
    simname = "danishfire"
    
    data = readcsv("output/danish.csv", header=true)[1]

    tt_ = Float64.(Dates.value.(Date.(data[:,1])))/365
    dt = tt_[2]-tt_[1]
    tt_ -= (tt_[1] - dt)
    prepend!(tt_, 0.0)

    dxx = Float64.(data[:,2])
  
    
    n = length(tt_)-1
    T = tt_[end]
    tt = tt_

    beta0 = 1/dt*1.812 # ml 
    alpha0 = 0.588942 

end

beta = beta0


P0a = GammaProcess(beta0a, alpha0a)
dxxa = rand.(increment.(diff(tt), P0a))
#X0a = sample(tt, P0a) 
X0a = SamplePath(tt, Bridge.cumsum0(dxxa))


P0b = GammaProcess(beta0b, alpha0b)
#X0b = sample(tt, P0b) 
dxxb = rand.(increment.(diff(tt), P0b))
X0b = SamplePath(tt, Bridge.cumsum0(dxxb))

P0 = GammaProcess(beta0, alpha0)
dxx0 = rand.(increment.(diff(tt), P0))
X0 = SamplePath(tt, Bridge.cumsum0(dxx0))


dt = mean(diff(X0a.tt))
PD = increment(dt, P0a)

@test shape(PD) ≈ beta0a*dt
@test scale(PD) ≈ 1/alpha0a
@test beta0a*dt/alpha0a ≈ mean(increment(dt, P0a))
@test beta0a*dt/alpha0a^2 ≈ var(increment(dt, P0a))

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

inc(B::SamplePath) = B.tt[end] - B.tt[1], B.yy[end] - B.yy[1]


pdf(x, beta, alpha) = beta/x*exp(-alpha*x)


#lp(s, x, t, y, P::GammaProcess) = logpdf(increment(t-s, P), y-x)

function dlp(dt, dx, Po::GammaProcess, P::GammaProcess) 
    assert(P.λ ==  Po.λ)
    alpha, = dt*P.γ
    alphao = dt*Po.γ
    #log(y-x)*(alphao-1) - (y-x)*P.λ - lgamma(alphao) + log(P.λ)*alphao - (log(y-x)*(alpha-1) - (y-x)*P.λ - lgamma(alpha) + log(P.λ)*alpha)
    log(dx)*(alphao-alpha)  + lgamma(alpha) - lgamma(alphao) + log(P.λ)*(alphao - alpha)    
end

function lp2(s, x, t, y, P::GammaProcess) 
    alpha = (t-s)*P.γ
    log(y-x)*(alpha-1) - (y-x)*P.λ - lgamma(alpha) + log(P.λ)*alpha
end
#\frac{x^{\alpha-1} e^{-x/\theta}}{\Gamma(\alpha) \theta^\alpha},

function addprc(X, Y)
    yy = copy(X.yy)
    for i in 1:length(yy)
        yy[i] += Y.yy[last(searchsorted(Y.tt, X.tt[i]))]        
    end
    SamplePath(X.tt, yy)
 end       

function scalebridge!(yy, xx, u, v)
    assert(xx[1] == 0)
    yy[:] = (xx-xx[1]) .* ((v-u)/(xx[end] - xx[1])) .+ u
end


if sim == :gamma
    X = X0a
    dxx = dxxa
elseif sim == :sumgamma
    X = addprc(X0a, X0b)
    dxx = dxxa + dxxb
else
    X = SamplePath(tt, Bridge.cumsum0(dxx))
end


# grid points
# b = quantile(increment(dt, GammaProcess(beta0,alpha0)),(1:(N))/(N+1)) # theoretical
if simid != 3 && N >= 1
    b = cumsum(0.3:0.4:2.0)[1:N]
    #b = quantile(diff(X.yy), (N:2N-1)/(2N)) # first bin resembles 50% of emperical increment distributions.
elseif N == 0
    b = Float64[]
end

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
Pi4 = Uniform(0.1, 1000.0) # beta

assert(mean(Pi) ≈ epi)
assert(var(Pi) ≈ vpi)



lpi(alpha, theta, rho) = logpdf(Pi, alpha) + sum(logpdf.(Pi2, theta)) + sum(logpdf.(Pi3, rho))
lpi(alpha, theta, rho, beta) = logpdf(Pi, alpha) + sum(logpdf.(Pi2, theta)) + sum(logpdf.(Pi3, rho)) + logpdf(Pi4, beta)
#initialize 

#...

# alpha, beta = truth


B = Vector{typeof(X)}(n)
Z = Vector{typeof(X)}(n)
Bº = Vector{typeof(X)}(n)
Zº = Vector{typeof(X)}(n)

#Bº = Vector{typeof(X)}(n)
tt = X.tt
yy = X.yy




# Bookkeeping
mkpath(joinpath("output", simname))
try # save cp of this file as documentation
    cp(@__FILE__(), joinpath("output",simname,"$simname.jl"); remove_destination=true)
end

beps = 0.0
#beps = b[1]/5
#yy = diff(vcat(B...).yy)
#var(yy[yy.< b[1]/4])


alpha = alpha0
beta = beta0

alpha = alpha
if transdim
    beta = beta
end

#c = beta*(T/(n*m*alpha)) *(1-exp(-alpha*beps)) # compensator for small jumps
epsij = 1e-50

theta = zeros(N)
rho = zeros(N)

# prior chain step std
alphasigma = 0.15
thsigma = 0.05
rhosigma = 0.05*ones(N) # variance of rho1 = 0
if N > 0 && sim != :fire
    rhosigma[1] = 0.0
end
betasigma = 0.01

if simid == 3
    alphasigma = 0.075
    betasigma = 0.5
end    
if isdefined(:nomcmc) && nomcmc
    error("don't run: nomcmc == true")
end

open(joinpath("output", simname,"truth.txt"), "w") do f
    bn = join(["b$i" for i in 1:length(b)], " ")
    println(f, "N alpha0 beta0 T n $bn beps prior1 prior2 ") 
    println(f, N, " ", join(round.([alpha0, beta0, T, n, b... ],3)," "), " $beps \"$Pi\" \"$Pi2\" ") 
end

# initial augmentation


for i in 1:n
    P0 = GammaProcess(beta, alpha)
    Pº = GammaBridge(tt[i+1], yy[i+1], P0)
    tti = linspace(tt[i], tt[i+1], m+1)
    Z[i] = sample(tti, P0, 0.0)
    Zº[i] = copy(Z[i])

    B[i] = copy(Z[i])
    scalebridge!(B[i].yy, Z[i].yy, yy[i], yy[i+1])
    Bº[i] = copy(B[i])
end
# workspaces
zz = copy(B[1].yy)
bb = copy(B[1].yy)
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

mc = mcstart([alpha; beta; theta; rho])
thacc = 0
Bacc = 0
alphaacc = 0
betaacc = 0

for iter in 1:iterations
    # logging
    mc = mcnext(mc, [alpha; beta; theta; rho])
    open(joinpath("output", simname, "params.txt"), "a") do f
        println(f, iter, " ", join(round.([alpha; beta; theta; rho], 8), " ")) 
    end
   
    # compensator for small jumps
    #c = beta*(T/(n*m*alpha)) * (1 - exp(-alpha*beps))
    c = 0.0

    P0 = GammaProcess(beta, alpha)
    P = LocalGammaProcess(P0, theta, rho, b)
    # sample bridges
    for i in 1:n
        #Pº = GammaBridge(tt[i+1], yy[i+1], P0)
        #Bº = SamplePath(B[i].tt, zz)
        #sample!(Bº, Pº, yy[i])
        sample!(Zº[i], P0, 0.0)
        scalebridge!(Bº[i].yy, Zº[i].yy, yy[i], yy[i+1])

        ll = llikelihood(Bº[i], P, c) - llikelihood(B[i], P, c) 
       # print(i, " ", ll, " ")
        if rand() < exp(ll)
            B[i], Bº[i] = Bº[i], B[i]
            Z[i], Zº[i] = Zº[i], Z[i]
            Bacc += 1
        end
    end

    # sample parameters
    # update theta and rho
    if iter % 5 != 2 # remember to update formula for acceptane rates
        P0 = GammaProcess(beta, alpha)
        thetaº = theta + thsigma*randn(length(theta))
        rhoº = rho + rhosigma.*randn(length(rho))
        
        if N == 0 || thetaº[end] + alpha < eps() 
            # reject
        else
            P = LocalGammaProcess(P0, theta, rho, b)
            Pº = LocalGammaProcess(P0, thetaº, rhoº, b)

            ll = 0.0
            for i in 1:n
                ll += llikelihood(B[i], Pº, P, c)
            end
            print("$iter \t\t\t\t paramsº: ", round(ll, 5), " ", round.([thetaº; rhoº], 3))
            if rand() < exp(ll + lpi(alpha, thetaº, rhoº, beta) - lpi(alpha, theta, rho, beta))
                print("✓")
                theta = thetaº
                rho = rhoº
                thacc += 1
            end
            println()
        end
    end

    if transdim && iter % 5 == 4
        betaº = beta + betasigma*randn()
        if betaº <= 0
            #reject
        else
            ll = 0.0
            P0 = GammaProcess(beta, alpha)
            P0º = GammaProcess(betaº, alpha)
            P = LocalGammaProcess(P0, theta, rho, b)
            Pº = LocalGammaProcess(P0º, theta, rho, b)
        
            if betaº >= beta
                PD = GammaProcess(betaº - beta, alpha)

                for i in 1:n
                    sample!(Zº[i], PD, 0.0)
                    Zº[i].yy[:] += Z[i].yy

                    scalebridge!(Bº[i].yy, Zº[i].yy, yy[i], yy[i+1])

                    ll += llikelihood(Bº[i], Pº, c) - llikelihood(B[i], P, c) + dlp(tt[i+1] - tt[i], dxx[i], P0º, P0)
                end
            else # thinning
                for i in 1:n
                    
                    Zº[i].yy[1] = 0.0
                    dtti = step(linspace(tt[i], tt[i+1], m+1))
                    

                    Be = Beta(dtti*betaº, dtti*(beta-betaº))
                    for j in 2:length(Zº[i].tt)
                        #dt = Z[i].tt[j] - Z[i].tt[j-1]
                        #Be = Beta(dt*betaº, dt*(beta-betaº))
                        Zº[i].yy[j] = Zº[i].yy[j-1] + (Z[i].yy[j] - Z[i].yy[j-1])*rand(Be)
                    end
                   
                    scalebridge!(Bº[i].yy, Zº[i].yy, yy[i], yy[i+1])
                    ll += llikelihood(Bº[i], Pº, c) - llikelihood(B[i], P, c) + dlp(tt[i+1] - tt[i], dxx[i], P0º, P0)
                end
                
            end     
            print("$iter \t ")
            print_with_color(:green, "betaº: ", round(ll, 5), " [", round(betaº, 3), "]")     
            #println(llº, " ", ll, " ", lpi(alpha, theta, rho, betaº), " ", lpi(alpha, theta, rho, beta))
            if rand() < exp(ll + lpi(alpha, theta, rho, betaº) - lpi(alpha, theta, rho, beta))
                for i in 1:n
                    B[i], Bº[i] = Bº[i], B[i]
                    Z[i], Zº[i] = Zº[i], Z[i]
                end
                beta = betaº
                betaacc += 1
                print_with_color(:green, "✓")
            end
            println()
        end
    end
    
    if  !fixalpha && iter % 5 == 2 # remember to update formula for acceptane rates
        alphaº = alpha + alphasigma*randn()
        
        if alphaº < 0 || (N > 0 && theta[end] + alphaº < eps()) # according to Wilkinson
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
            if rand() < exp(ll + lpi(alphaº, theta, rho, beta) - lpi(alpha, theta, rho, beta))
                alpha = alphaº
                alphaacc += 1
                print("✓")
            end
            println() 
        end  
    end

end    


println("alpha acc ", alphaacc/(iterations/5))
println("beta acc ", betaacc/(iterations/5))
println("theta acc ", thacc/(4iterations/5))
println("posterior band")
display(hcat(mcband(mc)...))
println("posterior band for mean")
display(hcat(mcbandmean(mc)...))

params = readdlm(joinpath("output", simname, "params.txt"), Float64; skipstart=1)[:,2:end];

function plotparams(simname = "gammap") 
    params = readdlm(joinpath("output", simname, "params.txt"), Float64; skipstart=1)[:,2:end];
    clf();
    plot(params[:,1], color=:blue , lw = 0.7)
    plot(params[:,2], color=:blue , lw = 0.7)

    plot(params[:,3:end], color=:lightblue , lw = 0.4)
    plot(runmean(params), color=:darkgrey, lw = 0.4)

    params
end

Ph = GammaProcess( mean(params,1)[2:-1:1]... )
dxxh = rand.(increment.(diff(tt), Ph))
Xh = SamplePath(tt, Bridge.cumsum0(dxxh))

