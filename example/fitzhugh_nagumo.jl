using Bridge, Distributions, StaticArrays
using ConjugatePriors

PLOT = :winston
include("plot.jl")

srand(10)
        
diag2(x, y) = SDiagonal(@SVector [x,y])

struct FitzHughNagumo  <: ContinuousTimeProcess{SVector{2,Float64}}
    α::Float64
    β::Float64 
    γ1::Float64
    γ2::Float64
    ϵ::Float64
    s::Float64
    σ::SDiagonal{2,Float64}
    a::SDiagonal{2,Float64}
#    σ::SMatrix{2,2,Float64}
#    a::SMatrix{2,2,Float64}

    FitzHughNagumo(σ, α = 1/3,  β = 0.08*0.7, γ1 = 0.08, γ2 = 0.08*0.8,  ϵ = 1.,  s = 1.) = new(α, β, γ1, γ2, ϵ, s, σ, σ*σ') #'
end
#phi(t, x, P::FitzHughNagumo) = (@SMatrix [-x[1]^3 x[1]-x[2] 1. 0. 0. 0. ; 0. 0. 0. x[1] -x[2] 1.])
phi(t, x, P::FitzHughNagumo) = (@SMatrix [0.  -x[1]^3 + x[1]-x[2] 1. ; x[1]  0.  0. ]) #γ, ϵ, s 
 
Bridge.b(t, x, P::FitzHughNagumo) = (@SVector [-P.α*x[1]^3+ P.ϵ*(x[1]-x[2]) + P.s, P.γ1*x[1]- P.γ2*x[2] + P.β])
Bridge.σ(t, x, P::FitzHughNagumo) = P.σ
Bridge.a(t, x, P::FitzHughNagumo) = P.a
Bridge.Γ(t, x, P::FitzHughNagumo) = inv(P.a)

function conjugateb(YY, th, xi, P)
        n = length(th)
        G = zeros(n,n)
        mu = zeros(th)
        for Y in YY
            for i in 1:length(Y)-1
                phii = phi(Y[i]..., P)
                Gphii = Bridge.Γ(Y[i]..., P)*phii
                zi = phii'*Gphii
                dy = Y.yy[i+1] - Y.yy[i]
                mu = mu + Vector(Gphii'*dy)
                G[:] = G +  Matrix(zi * (Y.tt[i+1]-Y.tt[i]))
            end
        end #for m
        WW = G .+ diagm(xi)
        WL = transpose(chol(WW))
        th° = WL'\(randn(n)+WL\mu)
end        

mcstart(yy) = (zeros(yy), zeros(yy), 0)
function mcnext(mc, x)  
    m, m2, n = mc
    delta = x - m
    n = n + 1
    m = m + delta*(1/n)
    m2 = m2 + map((x,y)->x.*y, delta, (x - m))
    m, m2, n
end 
function mcbandste(mc) 
    m, m2, k = mc
    Q = sqrt(2.)*erfinv(0.95)
    
    ste = eltype(m)[sqrt(v) for v in m2 * (1/(k - 1))]*sqrt(1/k)
    m -Q*ste, m +Q*ste
end
function mcband(mc) 
    m, m2, k = mc
    Q = sqrt(2.)*erfinv(0.95)
    
    std = eltype(m)[sqrt.(v) for v in m2 * (1/(k - 1))]
    m-Q*std, m+Q*std
end


function param(θ, σ)
    β, γ, ϵ, s = θ
    σ1,σ2 = σ
#   σ,    α, β, γ1,γ2, ϵ,  s 
    diag2(σ1,σ2), ϵ, β, γ, 1., ϵ, s 
end
simid = 2
simname =["exci", "nonexci"][simid]
mkpath(joinpath("output",simname))
try 
    cp(@__FILE__(), joinpath("output",simname,"$simname.jl"); remove_destination=true)
end

θ =  [[0.6, 1.4][simid], 1.5, 10., 0.5*10] # [β, γ, ϵ, s]
θtrue=copy(θ)
scaleθ = 0.08*[0., 1., 5., 1.]
σ = [0.3, 0.5] #[σ1]
scaleσ = [0.01]
σtrue = copy(σ)
Ptrue = FitzHughNagumo(param(θ, σ)...)
si = sitrue = 0.1 # observation error 



n = 100 # number of segments
m = 100 # number of euler steps per segments
TT = 20.
dt = TT/n/m
tt = linspace(0., TT, n*m+1)
tttrue = linspace(0., TT, n*10*m+1)
ttf = tt[1:m:end]

uu = (@SVector [0., 1.])
Y = euler(uu, sample(tttrue, Wiener{SVector{2,Float64}}()), Ptrue) 


Ytrue = Y[1:10m:end] #subsample
assert(endof(Ytrue) == n+1)

L = @SMatrix [ 1. 0. ]
Yobs = SamplePath{SVector{1,Float64}}(Ytrue.tt, [L*Ytrue.yy[i] + si*randn(SVector{1,Float64}) for i in 1:length(Ytrue.yy)])

r = [(:xrange,(-2,2)), (:yrange,(-1,3))]


    

if false
    r = [(:xrange,(-2,2)), (:yrange,(-1,3))]


    plot(Y, xrange=(-4, 4), yrange=(-4,4))
    plot(Yobs, "o:")
    plot(Yobs[1:2],"o:")

    i = 4
    v = SVector(Matrix(L)\Vector(Yobs.yy[i+1]))
    P° = PBridgeProp(Ptrue, Ytrue[i]..., Yobs.tt[i+1], v, Ytrue[i+2]..., L,si^2*I, Ptrue.a)
    B = bridge(sample(tt[m*(i-1)+1:m*(i+1)+1] , Wiener{SVector{2,Float64}}()),P°)
    # B = eulerb(sample(tt[m*(i-1)+1:m*(i+1)+1] , Wiener{SVector{2,Float64}}()),P°)
    plot(B; r...)
    oplot(Ytrue[i:i+2],"ro:"; r...)
    oplot([Yobs.yy[i+1][1],Yobs.yy[i+1][1]],[-2, 2], ":"; r...)
    oplot(Y[1:2m+1], "b"; r...)
end

#Y2 = SamplePath(tt, copy(Y.yy[1:10:end])) #initialize with truth
Y2 = SamplePath{SVector{2,Float64}}(tt, map(x->x+uu,zeros(SVector{2,Float64}, length(tt)))) #initialize path with constant
Yfil = Y2[1:m:end] 

θs = Float64[]
iter = 0
tt2 = [collect(tt[m*(i-1)+1:m*(i+1)+1]) for i in 1:n-1]
BB = SamplePath[Y2[m*(i-1)+1:m*i+1] for i in 1:n]
BBnew = SamplePath[Y2[m*(i-1)+1:m*i+1] for i in 1:n]

# arbitrary starting values
si = 0.3
σ = [0.7, 0.7]
θ = 0.0 + 0.0*rand(length(θ))


for i in 1:length(θ) # start with truth for parameters not to be estimated
    if scaleθ[i] == 0
        θ[i] = θtrue[i]
    end
end    

P = FitzHughNagumo(param(θ, σ)...)


if false
    for i in 1:n # reparametrization
        v = SVector(Matrix(L)\Vector(Yobs.yy[i+1]))
        P° = PBridgeProp(P, Ytrue[i]..., Yobs.tt[i+1], v, 1000., (@SVector [0.,0.]), L,  si^2*I, P.a)
        BB[i] = bridge(sample(tt[m*(i-1)+1:m*i+1], Wiener{SVector{2,Float64}}()),P°)
        # BB[i] = eulerb(sample(tt[m*(i-1)+1:m*i+1], Wiener{SVector{2,Float64}}()),P°)
        Ytrue.yy[i+1] = BB[i].yy[m+1]
    end
end
plot2(Y, "r","r" ; yrange=(-3,3),linewidth=0.5)
display(oplot2(Y2, "b", "b"; yrange=(-3,3), linewidth=0.7))
        
i = 1

# reserve some space
ww = Array{SVector{2,Float64},1}(length(m*(i-1)+1:m*(i+1)+1))
yy = copy(ww)

# Prior
Alpha = 1/100
Beta= 1/100

piσ²(s2) = pdf(InverseGamma(Alpha,Beta), s2[1])*pdf(InverseGamma(Alpha,Beta), s2[2])
lq(x, si) = Bridge.logpdfnormal(x, si^2*I)
PiError = InverseGamma(Alpha,Beta)

xi = 1./[20., 20., 20]


open(joinpath("output",simname,"truth.txt"), "w") do f
    println(f, "beta gamma eps s sigma err") 
    println(f, join(round.([θtrue ; σtrue; sitrue],3)," ")) 
end

open(joinpath("output",simname,"params.txt"), "w") do f
    println(f, "n beta gamma eps s sigma err") 
end

open(joinpath("output",simname,"info.txt"), "w") do f
    println(f, "n $n m $m T $TT A $Alpha B $Beta") 
    println(f, "Y0 = $uu") 
    println(f, "xi = $xi") 
end


bacc = 0

mc = mcstart(vcat([BB[i][1:end-1] for i in 1:n]...).yy )
mcparams = mcstart([θ ; σ; si])

Bridge.constdiff(::FitzHughNagumo) = true

while true
    P = FitzHughNagumo(param(θ, σ)...)
    iter += 1


    for j = 1:2

        for i in j:2:n-1
            v = SVector{2,Float64}(Matrix(L)\Vector(Yobs.yy[i+1]))
            cs = Bridge.CSpline(Yobs.tt[i], Yobs.tt[i+2], Bridge.b( Yfil[i]..., P),  Bridge.b( Yfil[i+2]..., P))
            
            P° = PBridgeProp(P, Yfil[i]..., Yobs.tt[i+1], v, Yfil[i+2]..., L,  si^2*I, P.a, cs)
            B2 = bridge!(SamplePath(tt2[i], yy), sample!(SamplePath(tt2[i],ww), Wiener{SVector{2,Float64}}()),P°)
            # B2 = eulerb!(SamplePath(tt2[i], yy), sample!(SamplePath(tt2[i],ww), Wiener{SVector{2,Float64}}()),P°)
            
            llold = llikelihood([BB[i]; BB[i+1][2:end]], P°) + lq(Yobs.yy[i+1] -L*Yfil.yy[i+1], si)
            llnew = llikelihood(B2, P°) + lq(Yobs.yy[i+1]-L*B2.yy[m+1], si)
         #   println("$i->$(i+2) ($llnew - $llold) ")
            if rand() < exp(llnew - llold) 
                bacc += 1
                Yfil.yy[i+1] = B2.yy[m+1]
                BB[i] = B2[1:m+1]
                BB[i+1] = B2[m+1:end]
            end
        end
 
    end
    let i = n #do the last bridge
            v = SVector{2,Float64}(Matrix(L)\Vector(Yobs.yy[i+1]))
            P° = PBridgeProp(P, Yfil[i]..., Yobs.tt[i+1], v, 1000., (@SVector [0.,0.]), L,  si^2*I, P.a)
            B2 = euler(P°.v0, sample(tt[m*(i-1)+1:m*i+1], Wiener{SVector{2,Float64}}()),P°)
            
            llold = llikelihood(BB[i], P°) + lq(Yobs.yy[i+1] -L*Yfil.yy[i+1], si)
            llnew = llikelihood(B2, P°) + lq(Yobs.yy[i+1]-L*B2.yy[m+1], si)
            #println("$i->$(i+1) ($llnew - $llold) ")
    
            if rand() < exp(llnew - llold)
                Yfil.yy[i+1] = B2.yy[m+1]
                BB[i] = B2
            end
    end

# update error variance
    if size(L,1) == 1 # one dimensional
        residual = Yobs.yy - map(x->L*x,Yfil.yy)
        si = sqrt(rand(ConjugatePriors.posterior_canon(PiError,suffstats(Distributions.NormalKnownMu(0.), Bridge.mat(residual)))))
        si = sqrt(rand(PiError))
    else
        error("todo: multivariate observations")
    end    
    
    # update theta

    if iter % 1 == 1
        BBall = vcat([BB[i][1:end-1] for i in 1:n]...)
        θ° = θ + (2rand(length(θ)) .- 1).*scaleθ
        Pθ = FitzHughNagumo(param(θ, σ)...)
        Pθ° = FitzHughNagumo(param(θ°, σ)...)
        ll = girsanov(BBall, Pθ°, Pθ)
    #    println(ll)
        
        if rand()<exp(ll)  #iter > 200 
            θ = θ°
        end
    end

    # update conjugate theta

    θ[2:end] = conjugateb(BB, θ[2:end],xi, P)
    
    
    # update sigma (and theta)
    if iter % 1 == 0
        σ° = σ .* exp.(scaleσ .* randn(length(σ))) 
        θ° = θ #+ (2rand(length(θ)) .- 1).*scaleθ/3
        Pσ = FitzHughNagumo(param(θ, σ)...)
        Pσ° = FitzHughNagumo(param(θ°, σ°)...)
        ll = 0.
        for i in 1:n # reparametrization
            cs = Bridge.CSpline(Yobs.tt[i], Yobs.tt[i+1], Bridge.b( Yfil[i]..., P),  Bridge.b( Yfil[i+1]..., P))
           
            P° = BridgeProp(Pσ, Yfil[i]..., Yfil[i+1]..., Pσ.a, cs)
            P°° = BridgeProp(Pσ°, Yfil[i]..., Yfil[i+1]..., Pσ°.a, cs)
            Z = innovations(BB[i], P°)

            BBnew[i] = bridge(Z, P°°)
            # BBnew[i] = eulerb(Z, P°°)
            ll += lptilde(P°°) - lptilde(P°) + llikelihood(BBnew[i], P°°) - llikelihood(BB[i], P°)
    #        println(lptilde(P°°), " - ", ptilde(lP°))
    #        println(llikelihood(BBnew[i], P°°), " - ",  llikelihood(BB[i], P°))
        end
        #print(ll)
        # f(σ²) = log σ²
        # df(σ²) = 1/σ²
        # 2 log σ° = 2 log σ + Z, Z ~ N(0,v)
        # q(σ°²|σ²) = 1/(σ°² sqrt(2piv)) exp(-1/2v |log(σ°²) - log(σ²)|^2)
        # q(σ²|σ°²) / q(σ°²|σ²) = σ°²/σ²
        
        if rand() < exp(ll) * (piσ²(σ°.^2)/piσ²(σ.^2)) * (prod(σ°.^2)/prod(σ.^2))
            σ = σ°
            θ = θ°
            for i in 1:n
                BB[i] = BBnew[i]
            end
        end                    
    end     
    open(joinpath("output",simname,"params.txt"), "a") do f; println(f, iter, " ", join(round.([θ ; σ; si],8)," ")) end
    println(iter, "\t", join(round.([θ./θtrue; σ./σtrue; si/sitrue; 100bacc/iter/n  ],3),"\t"))
      
    BBall = vcat([BB[i][1:end-1] for i in 1:n]...)  
      
    
    mc = mcnext(mc, BBall.yy)  
    mcparams = mcnext(mcparams, [θ ; σ; si])
      
    if iter % 10 == 0
        xr = (5,7)
        @show iter
        plot2(Y, "r","r" ; xrange=xr, yrange=(-3,3),linewidth=0.5)
        oplot2(Yfil, "o","o"; xrange=xr, yrange=(-3,3),linewidth=0.1)
        oplot(Yobs, "ro"; xrange=xr, yrange=(-3,3),linewidth=0.1) 
        display(oplot2(BBall, "b", "b"; xrange=xr, yrange=(-3,3), linewidth=0.7))
    end
    if iter >= 100 break end
end

plot2(Y, "r","b" ; yrange=(-3,3),linewidth=0.5)
oplot(Yobs,"+r")
savefig(joinpath("output",simname,"truth.svg"))
savefig(joinpath("output",simname,"truth.pdf"))

plot2(Y, "r","b" ; yrange=(-3,3),linewidth=0.5)
oplot(Yobs,"or";symbolsize=0.3)
mcb = mcband(mc);
oplot2(SamplePath(tt, mcb[1]),"r","b";linewidth=0.5)
oplot2(SamplePath(tt, mcb[2]),"r","b";linewidth=0.5)
hcat(mcbandste(mcparams)..., [θtrue; σtrue; si])

savefig(joinpath("output",simname,"band.svg"))
savefig(joinpath("output",simname,"band.pdf"))

plot2(vcat([BB[i] for i in 1:n]...), "r", "b"; yrange=(-3,3), linewidth=0.5)
oplot(Yobs,"+r")
savefig(joinpath("output",simname,"sample.svg"))
savefig(joinpath("output",simname,"sample.pdf"))
