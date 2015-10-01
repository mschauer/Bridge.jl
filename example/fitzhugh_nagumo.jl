using Winston, Bridge, Distributions, FixedSizeArrays

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


function oplot2(Y::SamplePath{Vec{2,Float64}}, a1, a2; keyargs...) 
    yy = Bridge.mat(Y.yy)
    oplot(Y.tt, yy[1,:], a1; keyargs...)
    oplot(Y.tt, yy[2,:], a2; keyargs...)
end    
function plot2(Y::SamplePath{Vec{2,Float64}}, a1, a2; keyargs...) 
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

    FitzHughNagumo(σ = 0.5I, α = 1/3,  β = 0.08*0.7, γ1 = 0.08, γ2 = 0.08*0.8,  ϵ = 1.,  s = 1.) = new(α, β, γ1, γ2, ϵ, s, σ, σ*σ')
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

β, γ, ϵ, s,  = 0.6, 1.5, 10., 0.5 
σ1 = 0.5
#β, γ, ϵ, s = 1.4, 1.5, 0.1, 0.5
function param(β, γ, ϵ, s, σ1)
#    σ,α,β,γ1,γ2,ϵ,s = 
#    Mat(((σ1, 0.), (0.,σ2))), 1/ϵ, β, γ, 1., 1/ϵ, 1/ϵ
    σ1*I, ϵ, β, γ, 1., ϵ, s*ϵ
end
thetab = [β, γ, ϵ, s]
thetabtrue=copy(thetab)
estthetab = [1., 1., 5., 1.]
thetaσ = [σ1]
thetaσtrue = copy(thetaσ)
Ptrue = FitzHughNagumo(param(thetab ..., thetaσ... )...)


n = 200 # number of segments
m = 50 # number of euler steps per segments
TT = 20.
dt = TT/n/m
tt = linspace(0., TT, n*m+1)
ttf = tt[1:m:end]

Y = euler(Vec(0., 1.), sample(tt, Wiener{Vec{2,Float64}}()), Ptrue); 


Yfil = Y[1:m:end] #subsample
Ytrue = copy(Yfil)
assert(endof(Yfil) == n+1)
si = 0.5 # observation error 

L = Mat(((1.,),(0.,)))
Yobs = SamplePath{Vec{1,Float64}}(Yfil.tt, [L*Yfil.yy[i] .+ si*randn(Vec{1,Float64}) for i in 1:length(Yfil.yy)])

r = [(:xrange,(-2,2)), (:yrange,(-1,3))]


    

if false
    plot(Y, xrange=(-4, 4), yrange=(-4,4))
    plot(Yobs, "o:")
    plot(Yobs[1:2],"o:")
    i = 1

    v = Vec(Matrix(L)\Vector(Yobs.yy[i+1]))
    P° = PBridgeProp(Ptrue, Yfil[i]..., Yobs.tt[i+1], v, Yfil[i+2]..., L,si^2*I, Ptrue.a)
    B = euler(P°.v0, sample(tt[m*(i-1)+1:m*(i+1)+1] , Wiener{Vec{2,Float64}}()),P°)
    plot(B; r...)
    oplot(Yfil[i:i+2],"ro:"; r...)
    oplot([Yobs.yy[i+1][1],Yobs.yy[i+1][1]],[-2, 2], ":"; r...)
    oplot(Y[1:2m+1], "b"; r...)
end

#Y2 = euler(Y.yy[1], sample(tt, Wiener{Vec{2,Float64}}()), Ptrue); 
#Y2 = SamplePath(Y.tt, copy(Y.yy) .+ Vec(0., 0.1))
Y2 = SamplePath{Vec{2,Float64}}(tt, zeros(Vec{2,Float64}, length(tt)))

iter = 0
tt2 = [collect(tt[m*(i-1)+1:m*(i+1)+1]) for i in 1:n-1]
BB = SamplePath[Y2[m*(i-1)+1:m*i+1] for i in 1:n]
BBnew = SamplePath[Y2[m*(i-1)+1:m*i+1] for i in 1:n]

thetaσ = [0.7]
thetab = 0.0 + 0.0*rand(length(thetab))

P = FitzHughNagumo(param(thetab..., thetaσ...)...)


if false
    for i in 1:n # reparametrization
        v = Vec(Matrix(L)\Vector(Yobs.yy[i+1]))
        P° = PBridgeProp(P, Yfil[i]..., Yobs.tt[i+1], v, 1000., Vec(0.,0.), L,  si^2*I, P.a)
        BB[i] = euler(P°.v0, sample(tt[m*(i-1)+1:m*i+1], Wiener{Vec{2,Float64}}()),P°)
        Yfil.yy[i+1] = BB[i].yy[m+1]
    end
end
        plot2(Y, "r","r" ; yrange=(-3,3),linewidth=0.5)
        display(oplot2(Y2, "b", "b"; yrange=(-3,3), linewidth=0.7))
        
i = 1

# reserve some space
ww = Array{Vec{2,Float64},1}(length(m*(i-1)+1:m*(i+1)+1))
yy = copy(ww)


lq(x) = Bridge.logpdfnormal(x, si^2*I)

while true
    P = FitzHughNagumo(param(thetab..., thetaσ...)...)
    iter += 1
 

    for j = 1:2
        for i in j:2:n-1

            v = Vec(Matrix(L)\Vector(Yobs.yy[i+1]))
            P° = PBridgeProp(P, Yfil[i]..., Yobs.tt[i+1], v, Yfil[i+2]..., L,  si^2*I, P.a)
            B2 = euler!(SamplePath(tt2[i], yy), P°.v0, sample!(SamplePath(tt2[i],ww), Wiener{Vec{2,Float64}}()),P°)
            
            llold = llikelihood([BB[i]; BB[i+1][2:end]], P°) + lq(Yobs.yy[i+1] -L*Yfil.yy[i+1])
            llnew = llikelihood(B2, P°) + lq(Yobs.yy[i+1]-L*B2.yy[m+1])
            #println("$i->$(i+2) ($llnew - $llold) ")
            if rand() < exp(llnew - llold) 
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
            
            llold = llikelihood(BB[i], P°) + lq(Yobs.yy[i+1] -L*Yfil.yy[i+1])
            llnew = llikelihood(B2, P°) + lq(Yobs.yy[i+1]-L*B2.yy[m+1])
            #println("$i->$(i+1) ($llnew - $llold) ")
    
            if rand() < exp(llnew - llold)
                Yfil.yy[i+1] = B2.yy[m+1]
                BB[i] = B2
            end
    end
    
# update parameter (except sigma)

if iter % 1 == 0
    BBall = vcat([BB[i][1:end-1] for i in 1:n]...)
    thetab° = thetab + 0.1*(2rand(length(thetab)) .- 1).*estthetab
    Pb = FitzHughNagumo(param(thetab..., thetaσ...)...)
    Pb° = FitzHughNagumo(param(thetab°..., thetaσ...)...)
    ll = girsanov(BBall, Pb°, Pb)
#    println(ll)
    
    if rand()<exp(ll)  #iter > 200 
        thetab = thetab°
    end
end
# update sigma
if iter % 1 == 0
    thetaσ° = thetaσ .* exp(0.03(2rand(length(thetaσ)) .- 1))
    
    Pσ = FitzHughNagumo(param(thetab..., thetaσ...)...)
    Pσ° = FitzHughNagumo(param(thetab..., thetaσ°...)...)
    ll = 0.
    for i in 1:n # reparametrization
        P° = BridgeProp(Pσ, Yfil[i]..., Yfil[i+1]..., Pσ.a)
        P°° = BridgeProp(Pσ°, Yfil[i]..., Yfil[i+1]..., Pσ°.a)
        Z = innovations(BB[i], P°)

        BBnew[i] = euler(P°°.v0, Z, P°°)
        ll += ptilde(P°°) - ptilde(P°) + llikelihood(BBnew[i], P°°) - llikelihood(BB[i], P°)
#        println(ptilde(P°°), " - ", ptilde(P°))
#        println(llikelihood(BBnew[i], P°°), " - ",  llikelihood(BB[i], P°))
    end
    if rand() < exp(ll)
        thetaσ = thetaσ°
        for i in 1:n
            BB[i] = BBnew[i]
        end
    end                    
end     
    println(iter, " ", round([thetab./thetabtrue; thetaσ./thetaσtrue],3))
      
    if iter % 10 == 0
        BBall = vcat([BB[i][1:end-1] for i in 1:n]...)
        plot2(Y, "r","r" ; yrange=(-3,3),linewidth=0.5)
        display(oplot2(BBall, "b", "b"; yrange=(-3,3), linewidth=0.7))
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
