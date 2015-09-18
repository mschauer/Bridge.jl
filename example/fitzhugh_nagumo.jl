using Winston, Bridge, Distributions, FixedSizeArrays

import Winston: plot, oplot
function plot(Y::CTPath{Vec{2,Float64}}, args...; keyargs...) 
    yy = Bridge.mat(Y.yy)
    plot(yy[1,:], yy[2,:], args...; keyargs...)
end    
function plot(Y::CTPath{Vec{1,Float64}}, args...; keyargs...) 
    yy = Bridge.mat(Y.yy)
    plot(Y.tt, yy[1,:], args...; keyargs...)
end    
function plot(Y::CTPath{Float64}, args...; keyargs...) 
    plot(Y.tt, Y.yy, args...; keyargs...)
end    

function oplot(Y::CTPath{Vec{1,Float64}}, args...; keyargs...) 
    yy = Bridge.mat(Y.yy)
    oplot(Y.tt, yy[1,:], args...; keyargs...)
end   
function oplot(Y::CTPath{Vec{2,Float64}}, args...; keyargs...) 
    yy = Bridge.mat(Y.yy)
    oplot(yy[1,:], yy[2,:], args...; keyargs...)
end    
function oplot(Y::CTPath{Float64}, args...; keyargs...) 
    oplot(Y.tt, Y.yy, args...; keyargs...)
end    


function oplot2(Y::CTPath{Vec{2,Float64}}, a1, a2; keyargs...) 
    yy = Bridge.mat(Y.yy)
    oplot(Y.tt, yy[1,:], a1; keyargs...)
    oplot(Y.tt, yy[2,:], a2; keyargs...)
end    
function plot2(Y::CTPath{Vec{2,Float64}}, a1, a2; keyargs...) 
    yy = Bridge.mat(Y.yy)
    plot(Y.tt,  yy[1,:], a1; keyargs...)
    oplot(Y.tt, yy[2,:], a2; keyargs...)
end    
        

    

srand(10)

 


immutable FitzHughNagumo  <: CTPro{Vec{2,Float64}}
    α::Float64
    β::Float64 
    γ1::Float64
    γ2::Float64
    ϵ::Float64
    s::Float64
    σ::Base.LinAlg.UniformScaling{Float64}
    a::Base.LinAlg.UniformScaling{Float64}
    FitzHughNagumo(σ = 0.5I, α = 1/3,  β = 0.08*0.7, γ1 = 0.08, γ2 = 0.08*0.8,  ϵ = 1.,  s = 1.) = new(α, β, γ1, γ2, ϵ, s, σ, σ*σ')
end
 
Bridge.b(t, x, P::FitzHughNagumo) = Vec(-P.α*x[1]^3+ P.ϵ*(x[1]-x[2]) + P.s, P.γ1*x[1]- P.γ2*x[2] +P.β)
Bridge.σ(t, x, P::FitzHughNagumo) = P.σ
Bridge.a(t, x, P::FitzHughNagumo) = P.a


P = FitzHughNagumo(0.2*I)
n = 50
m = 200
TT = 100.
dt = TT/n/m
tt = linspace(0., TT, n*m+1)
ttf = tt[1:m:end]

Y = euler(Vec(0., 1.), sample(tt, Wiener{Vec{2,Float64}}()), P); 


Yfil = Y[1:m:end] #subsample
Ytrue = copy(Yfil)
assert(endof(Yfil) == n+1)
si = 0.5 # observation error 

L = Mat(((1.,),(0.,)))
Yobs = CTPath{Vec{1,Float64}}(Yfil.tt, [L*Yfil.yy[i] .+ si*randn(Vec{1,Float64}) for i in 1:length(Yfil.yy)])

r = [(:xrange,(-2,2)), (:yrange,(-1,3))]


    

if false
    plot(Y, xrange=(-4, 4), yrange=(-4,4))
    plot(Yobs, "o:")
    plot(Yobs[1:2],"o:")
    i = 1

    v = Vec(Matrix(L)\Vector(Yobs.yy[i+1]))
    Pprop = PBridgeProp(P, Yfil[i]..., Yobs.tt[i+1], v, Yfil[i+2]..., L,si^2*I, P.a)
    B = euler(Pprop.v0, sample(tt[m*(i-1)+1:m*(i+1)+1] , Wiener{Vec{2,Float64}}()),Pprop)
    plot(B; r...)
    oplot(Yfil[i:i+2],"ro:"; r...)
    oplot([Yobs.yy[i+1][1],Yobs.yy[i+1][1]],[-2, 2], ":"; r...)
    oplot(Y[1:2m+1], "b"; r...)
end

Y2 = euler(Vec(0., 1.), sample(tt, Wiener{Vec{2,Float64}}()), P); 


iter = 0
tt2 = [collect(tt[m*(i-1)+1:m*(i+1)+1]) for i in 1:n-1]
BB = CTPath[Y2[m*(i-1)+1:m*i+1] for i in 1:n]
i = 1

# reserve some space
ww = Array{Vec{2,Float64},1}(length(m*(i-1)+1:m*(i+1)+1))
yy = copy(ww)




while true


    iter += 1
    println(iter)

    for j = 1:2
        for i in j:2:n-1

            v = Vec(Matrix(L)\Vector(Yobs.yy[i+1]))
            Pprop = PBridgeProp(P, Yfil[i]..., Yobs.tt[i+1], v, Yfil[i+2]..., L,  si^2*I, P.a)
            B2 = euler!(CTPath(tt2[i], yy), Pprop.v0, sample!(CTPath(tt2[i],ww), Wiener{Vec{2,Float64}}()),Pprop)
            
            llold = llikelihood([BB[i]; BB[i+1][2:end]], Pprop)
            llnew = llikelihood(B2, Pprop)
            println("$i->$(i+2) ($llnew - $llold) ")
            if (iter < 500 && llnew > 0) || exp(llnew - llold) > rand() 
                Yfil.yy[i+1] = B2.yy[m+1]
                BB[i] = B2[1:m+1]
                BB[i+1] = B2[m+1:end]
            end
        end
 
    end
    let i = n #do the last bridge
       v = Vec(Matrix(L)\Vector(Yobs.yy[i+1]))
            Pprop = PBridgeProp(P, Yfil[i]..., Yobs.tt[i+1], v, 1000., Vec(0.,0.), L,  si^2*I, P.a)
            B2 = euler(Pprop.v0, sample(tt[m*(i-1)+1:m*i+1], Wiener{Vec{2,Float64}}()),Pprop)
            
            llold = llikelihood(BB[i], Pprop)
            llnew = llikelihood(B2, Pprop)
            println("$i->$(i+1) ($llnew - $llold) ")
    
            if (iter < 500 && llnew > 0) || exp(llnew - llold) > rand() 
                Yfil.yy[i+1] = B2.yy[m+1]
                BB[i] = B2
            end
    end
    if iter % 10 == 0
        plot2(Y, "r","r" ; yrange=(-3,3),linewidth=0.5)
        display(oplot2(vcat([BB[i] for i in 1:n]...), "b", "b"; yrange=(-3,3), linewidth=0.7))
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
