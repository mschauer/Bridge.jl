using Winston, Bridge, Distributions, FixedSizeArrays, ConjugatePriors

import Winston: plot, oplot
function plot(Y::SamplePath{Vec{2,Float64}}, args...; keyargs...) 
    yy = Bridge.mat(Y.yy)
    plot(yy[1,:], yy[2,:], args...; keyargs...)
end    
function plot(Y::SamplePath{Vec{1,Float64}}, args...; keyargs...) 
    yy = Bridge.mat(Y.yy)
    plot(Y.tt, yy[1,:], args...; keyargs...)
end    
function plot(Y::SamplePath{Float64}, args...; keyargs...) 
    plot(Y.tt, Y.yy, args...; keyargs...)
end    

function oplot(Y::SamplePath{Vec{1,Float64}}, args...; keyargs...) 
    yy = Bridge.mat(Y.yy)
    oplot(Y.tt, yy[1,:], args...; keyargs...)
end   
function oplot(Y::SamplePath{Vec{2,Float64}}, args...; keyargs...) 
    yy = Bridge.mat(Y.yy)
    oplot(yy[1,:], yy[2,:], args...; keyargs...)
end    
function oplot(Y::SamplePath{Float64}, args...; keyargs...) 
    oplot(Y.tt, Y.yy, args...; keyargs...)
end    


function oplot2(Y::SamplePath{Vec{2,Float64}},  a1="r", a2="b";  keyargs...) 
    yy = Bridge.mat(Y.yy)
    oplot(Y.tt, yy[1,:], a1; keyargs...)
    oplot(Y.tt, yy[2,:], a2; keyargs...)
end    
function plot2(Y::SamplePath{Vec{2,Float64}}, a1="r", a2="b"; keyargs...) 
    yy = Bridge.mat(Y.yy)
    plot(Y.tt,  yy[1,:], a1; keyargs...)
    oplot(Y.tt, yy[2,:], a2; keyargs...)
end    
        

    

srand(10)

 


immutable FitzHughNagumo  <: ContinuousTimeProcess{Vec{2,Float64}}
    α::Float64
    β::Float64 
    γ1::Float64
    γ2::Float64
    ϵ::Float64
    s::Float64
    σ::Base.LinAlg.UniformScaling{Float64}
    a::Base.LinAlg.UniformScaling{Float64}
#    σ::FixedSizeArrays.Mat{2,2,Float64}
#    a::FixedSizeArrays.Mat{2,2,Float64}

    FitzHughNagumo(σ = 0.5I, α = 1/3,  β = 0.08*0.7, γ1 = 0.08, γ2 = 0.08*0.8,  ϵ = 1.,  s = 1.) = new(α, β, γ1, γ2, ϵ, s, σ, σ*σ') #'
end
 
Bridge.b(t, x, P::FitzHughNagumo) = Vec(-P.α*x[1]^3+ P.ϵ*(x[1]-x[2]) + P.s, P.γ1*x[1]- P.γ2*x[2] + P.β)
Bridge.σ(t, x, P::FitzHughNagumo) = P.σ
Bridge.a(t, x, P::FitzHughNagumo) = P.a
Bridge.Γ(t, x, P::FitzHughNagumo) = inv(P.a)





#σ = 0.2I
#α = 1/3
#β = 0.08*0.7
#γ1 = 0.08
#γ2 = 0.08*0.8
#ϵ = 1. 
#s = 1.
#estparam=[true,true, true, true, false, false]
#Ptrue = FitzHughNagumo(σ,α,β,γ1,γ2,ϵ,s)

function param(θ, σ)
    β, γ, ϵ, s = θ
    σ1 = σ[1]
#   σ,    α, β, γ1,γ2, ϵ,  s 
    σ1*I, ϵ, β, γ, 1., ϵ, s*ϵ #  Mat(((σ1, 0.), (0.,σ2)))
end
θ =  [[0.6, 1.4][1], 1.5, 10., 0.5] # [β, γ, ϵ, s]
θtrue=copy(θ)
scaleθ = 0.08*[0., 1., 5., 1.]
σ = [0.5] #[σ1]
scaleσ = [0.01]
σtrue = copy(σ)
Ptrue = FitzHughNagumo(param(θ, σ)...)
si = sitrue = 0.2 # observation error 



n = 150 # number of segments
m = 50 # number of euler steps per segments
TT = 30.
dt = TT/n/m
tt = linspace(0., TT, n*m+1)
tttrue = linspace(0., TT, n*10*m+1)
ttf = tt[1:m:end]

uu = Vec(0., 1.)
Y = euler(uu, sample(tttrue, Wiener{Vec{2,Float64}}()), Ptrue) 


Ytrue = Y[1:10m:end] #subsample
assert(endof(Ytrue) == n+1)

L = Mat(((1.,),(0.,)))
Yobs = SamplePath{Vec{1,Float64}}(Ytrue.tt, [L*Ytrue.yy[i] + si*randn(Vec{1,Float64}) for i in 1:length(Ytrue.yy)])

r = [(:xrange,(-2,2)), (:yrange,(-1,3))]


    

if false
    plot(Y, xrange=(-4, 4), yrange=(-4,4))
    plot(Yobs, "o:")
    plot(Yobs[1:2],"o:")
    i = 1

    v = Vec(Matrix(L)\Vector(Yobs.yy[i+1]))
    P° = PBridgeProp(Ptrue, Ytrue[i]..., Yobs.tt[i+1], v, Ytrue[i+2]..., L,si^2*I, Ptrue.a)
    B = eulerb(sample(tt[m*(i-1)+1:m*(i+1)+1] , Wiener{Vec{2,Float64}}()),P°)
    plot(B; r...)
    oplot(Ytrue[i:i+2],"ro:"; r...)
    oplot([Yobs.yy[i+1][1],Yobs.yy[i+1][1]],[-2, 2], ":"; r...)
    oplot(Y[1:2m+1], "b"; r...)
end

#Y2 = SamplePath(tt, copy(Y.yy[1:10:end])) #initialize with truth
Y2 = SamplePath{Vec{2,Float64}}(tt, zeros(Vec{2,Float64}, length(tt))+uu) #initialize path with constant
Yfil = Y2[1:m:end] 


iter = 0
tt2 = [collect(tt[m*(i-1)+1:m*(i+1)+1]) for i in 1:n-1]
BB = SamplePath[Y2[m*(i-1)+1:m*i+1] for i in 1:n]
BBnew = SamplePath[Y2[m*(i-1)+1:m*i+1] for i in 1:n]

# arbitrary starting values
si = 0.3
σ = [0.7]
θ = 0.0 + 0.0*rand(length(θ))


for i in 1:length(θ) # start with truth for parameters not to be estimated
    if scaleθ[i] == 0
        θ[i] = θtrue[i]
    end
end    

P = FitzHughNagumo(param(θ, σ)...)


if false
    for i in 1:n # reparametrization
        v = Vec(Matrix(L)\Vector(Yobs.yy[i+1]))
        P° = PBridgeProp(P, Ytrue[i]..., Yobs.tt[i+1], v, 1000., Vec(0.,0.), L,  si^2*I, P.a)
        BB[i] = eulerb(sample(tt[m*(i-1)+1:m*i+1], Wiener{Vec{2,Float64}}()),P°)
        Ytrue.yy[i+1] = BB[i].yy[m+1]
    end
end
plot2(Y, "r","r" ; yrange=(-3,3),linewidth=0.5)
display(oplot2(Y2, "b", "b"; yrange=(-3,3), linewidth=0.7))
        
i = 1

# reserve some space
ww = Array{Vec{2,Float64},1}(length(m*(i-1)+1:m*(i+1)+1))
yy = copy(ww)

# Prior
piσ²(s2) = pdf(InverseGamma(1/100,1/100), s2[1])
#piσ²(s2) = 1.#(max(0,(5000-10000s2[1])))^2

lq(x, si) = Bridge.logpdfnormal(x, si^2*I)
PiError = InverseGamma(1/100,1/100)

open("log.txt", "w") do f; println(f, 0, " ", join(round([θtrue ; σtrue; sitrue],3)," ")) end

bacc = 0

while true
    P = FitzHughNagumo(param(θ, σ)...)
    iter += 1


    for j = 1:2

        for i in j:2:n-1
            v = Vec(Matrix(L)\Vector(Yobs.yy[i+1]))
            cs = Bridge.CSpline(Yobs.tt[i], Yobs.tt[i+2], 0.7Bridge.b( Yfil[i]..., P),  0.7Bridge.b( Yfil[i+2]..., P))
            
            P° = PBridgeProp(P, Yfil[i]..., Yobs.tt[i+1], v, Yfil[i+2]..., L,  si^2*I, P.a, cs)
            B2 = eulerb!(SamplePath(tt2[i], yy), sample!(SamplePath(tt2[i],ww), Wiener{Vec{2,Float64}}()),P°)
            
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
            v = Vec(Matrix(L)\Vector(Yobs.yy[i+1]))
            P° = PBridgeProp(P, Yfil[i]..., Yobs.tt[i+1], v, 1000., Vec(0.,0.), L,  si^2*I, P.a)
            B2 = euler(P°.v0, sample(tt[m*(i-1)+1:m*i+1], Wiener{Vec{2,Float64}}()),P°)
            
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
    else
        error("todo: multivariate observations")
    end    
    
# update theta

    if iter % 1 == 0
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

    # update sigma (and theta)
    if iter % 1 == 0
        σ° = σ .* exp(scaleσ .* randn(length(σ))) 
        θ° = θ + (2rand(length(θ)) .- 1).*scaleθ/3
        Pσ = FitzHughNagumo(param(θ, σ)...)
        Pσ° = FitzHughNagumo(param(θ°, σ°)...)
        ll = 0.
        for i in 1:n # reparametrization
            cs = Bridge.CSpline(Yobs.tt[i], Yobs.tt[i+1], Bridge.b( Yfil[i]..., P),  Bridge.b( Yfil[i+1]..., P))
           
            P° = BridgeProp(Pσ, Yfil[i]..., Yfil[i+1]..., Pσ.a, cs)
            P°° = BridgeProp(Pσ°, Yfil[i]..., Yfil[i+1]..., Pσ°.a, cs)
            Z = innovations(BB[i], P°)

            BBnew[i] = eulerb(Z, P°°)
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
    open("log.txt", "a") do f; println(f, iter, " ", join(round([θ ; σ; si],3)," ")) end
    println(iter, "\t", join(round([θ./θtrue; σ./σtrue; si/sitrue; sqrt(σ[1]^2*TT/n + si[1])/sqrt(σtrue[1]^2*TT/n + sitrue[1]); 100bacc/iter/n  ],3),"\t"))
      
    if iter % 10 == 0
        xr = (5,7)
        BBall = vcat([BB[i][1:end-1] for i in 1:n]...)
        plot2(Y, "r","r" ; xrange=xr, yrange=(-3,3),linewidth=0.5)
        oplot2(Yfil, "o","o"; xrange=xr, yrange=(-3,3),linewidth=0.1)
        oplot(Yobs, "ro"; xrange=xr, yrange=(-3,3),linewidth=0.1) 
        display(oplot2(BBall, "b", "b"; xrange=xr, yrange=(-3,3), linewidth=0.7))
    end
    #if iter > 1000 break end
end


plot2(Y, "r","b" ; yrange=(-3,3),linewidth=0.5)
oplot(Yobs,"+r")
savefig("truth.svg")
savefig("truth.pdf")


plot2(vcat([BB[i] for i in 1:n]...), "r", "b"; yrange=(-3,3), linewidth=0.5)
oplot(Yobs,"+r")
savefig("sample.svg")
savefig("sample.pdf")
