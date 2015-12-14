using Bridge, Distributions, FixedSizeArrays, ConjugatePriors
PLOT = true

if PLOT
using Winston
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
end

#diag2(x, y) = FixedDiagonal(Vec([x,y]))
diag2(x, y) = Mat((x,0.),(0.,y))


# 2: 1655 seconds


immutable FitzHughNagumo  <: ContinuousTimeProcess{Vec{2,Float64}}
    α::Float64
    β::Float64 
    γ1::Float64
    γ2::Float64
    ϵ::Float64
    s::Float64
#    σ::FixedDiagonal{2,Float64}
#    a::FixedDiagonal{2,Float64}
    σ::FixedSizeArrays.Mat{2,2,Float64}
    a::FixedSizeArrays.Mat{2,2,Float64}

    FitzHughNagumo(σ, α = 1/3,  β = 0.08*0.7, γ1 = 0.08, γ2 = 0.08*0.8,  ϵ = 1.,  s = 1.) = new(α, β, γ1, γ2, ϵ, s, σ, σ*σ') #'
end


#Bridge.b(t, x, P::FitzHughNagumo) = Vec(-P.α*x[1]^3+ P.ϵ*(x[1]-x[2]) + P.s, P.γ1*x[1]- P.γ2*x[2] + P.β)
#phi234(t, x, P::FitzHughNagumo) = Mat((0., x[1]), (-x[1]^3 + x[1]-x[2], 0.), (1.,0.) ) #γ, ϵ, s 
function param(θ, σ)
    β, γ, ϵ, s = θ
    σ1, σ2 = σ
#   σ,    α, β, γ1,γ2, ϵ,  s 
    diag2(σ1,σ2), ϵ, β, γ, 1., ϵ, s  # α == ϵ
end
 
Bridge.b(t, x, P::FitzHughNagumo) = Vec(-P.α*x[1]^3 + P.ϵ*(x[1]-x[2]) + P.s, P.γ1*x[1]- P.γ2*x[2] + P.β)
phi1234(t, x, P::FitzHughNagumo) = Mat((0., 1.), (0., x[1]), (-x[1]^3 + x[1]-x[2], 0.),  (1.,0.) ) #β, γ, ϵ, s
intercept1234(t, x, P::FitzHughNagumo) = Vec(0, - P.γ2*x[2])

Bridge.σ(t, x, P::FitzHughNagumo) = P.σ
Bridge.a(t, x, P::FitzHughNagumo) = P.a
Bridge.Γ(t, x, P::FitzHughNagumo) = inv(P.a)


function conjugateb(YY, th, xi, P, phif, intc)
        n = length(th)
        G = zeros(n,n)
        mu = zeros(th)
        for Y in YY
            for i in 1:length(Y)-1
                phi = phif(Y[i]..., P)
                Gphi = Bridge.Γ(Y[i]..., P)*phi
                zi = phi'*Gphi
                dy = Y.yy[i+1] - Y.yy[i] 
                ds = Y.tt[i+1] - Y.tt[i] 
                mu = mu + Vector(Gphi'*(dy - intc(Y[i]..., P)*ds))
                G[:] = G +  Matrix(zi * (Y.tt[i+1]-Y.tt[i]))
            end
        end #for m
        WW = G .+ diagm(xi)
        WL = chol(WW,Val{:L})
        th° = WL'\(randn(n)+WL\mu)
end        

mcstart(yy) = (zeros(yy), zeros(yy), 0)
function mcnext(mc, x)  
    m, m2, n = mc
    delta = x - m
    n = n + 1
    m = m + delta*(1/n)
    m2 = m2 + map(.*, delta, x - m)
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
    
    std = eltype(m)[sqrt(v) for v in m2 * (1/(k - 1))]
    m-Q*std, m+Q*std
end

function MyProp(u, v, P, proptype=:mbb)
 
    
    if proptype == :guip
        cs = Bridge.CSpline(u[1], v[1], Bridge.b(u..., P),  Bridge.b( v..., P))
        return BridgeProp(P, u..., v..., P.a, cs)
    else
        return DHBridgeProp(P, u..., v...)
    end
end



############## Configuration ###################################
srand(10)
K = 100000


simid = 1
#propid = 1
proptype = [:mbb,:guip][propid]

simname =["full", "fullne"][simid] * "$proptype"


θ =  [[0.6, 1.4][simid], 1.5, 10., 0.5*10] # [β, γ, ϵ, s] 
θtrue=copy(θ)
scaleθ = 0.08*[1., 1., 5., 5.]
σ = [0.25, 0.2]
scaleσ = ([0.1, 0.1],[0.1, 0.1])[propid]

σtrue = copy(σ)
Ptrue = FitzHughNagumo(param(θ, σ)...)
 


n = 400 # number of segments
m = 200 # number of euler steps per segments
mextra = 20 #factor of extra steps for truth
TT = 300.
dt = TT/n/m
tt = linspace(0., TT, n*m+1)
tttrue = linspace(0., TT, n*mextra*m+1)
ttf = tt[1:m:end]

uu = Vec(0., 1.)

r = [(:xrange,(-2,2)), (:yrange,(-1,3))]

#######################################################

Y = euler(uu, sample(tttrue, Wiener{Vec{2,Float64}}()), Ptrue) 
Yobs = Y[1:mextra*m:end] #subsample
assert(endof(Yobs) == n+1)


Y2 = SamplePath(tt, copy(Y.yy[1:mextra:end])) #subsample 
normalize(tt) = (tt - tt[1])/(tt[end]-tt[1])

θs = Float64[]
iter = 0
tts = [collect(tt[m*(i-1)+1:m*(i)+1]) for i in 1:n]
BB = SamplePath[Y2[m*(i-1)+1:m*i+1] for i in 1:n]
for i in 1:n
    BB[i].yy[:] =  map(x -> BB[i].yy[1] + x*(BB[i].yy[end] - BB[i].yy[1]),normalize(tts[i]))
end
    
BBnew = copy(BB)
BBall = vcat([BB[i][1:end-1] for i in 1:n]...)

# 
if PLOT
        xr = (5,7)
        plot2(Yobs, "o", "o"; xrange=xr, yrange=(-3,3),linewidth=0.1) 
        display(oplot2(BBall, "b", "b"; xrange=xr, yrange=(-3,3), linewidth=0.7))
end



################### Prior ###################################
conjθs = [1,2,3,4] #set of conjugate thetas
phi = phi1234
intercept = intercept1234
estθ = [true, true, true, true] #params to estimate

# arbitrary starting values
 
σ = [0.7, 0.7]
θ = 0.5 + 1.0*rand(length(θ)) #θ = copy(θtrue)

for i in 1:length(θ) # start with truth for parameters not to be estimated
    if !estθ[i]
        θ[i] = θtrue[i]
        scaleθ[i] = 0.
    end
end    

P = FitzHughNagumo(param(θ, σ)...)

#if PLOT
#    plot2(Y, "r","r" ; yrange=(-3,3),linewidth=0.5)
#    display(oplot2(Y2, "b", "b"; yrange=(-3,3), linewidth=0.7))
#end
    


# reserve some space
i = 1
ww = Array{Vec{2,Float64},1}(length(m*(i-1)+1:m*(i+1)+1))
yy = copy(ww)

# Prior
Alpha = 1/500
Beta= 1/500

piσ²(s2) = pdf(InverseGamma(Alpha,Beta), s2[1])*pdf(InverseGamma(Alpha,Beta), s2[2])
lq(x, si) = Bridge.logpdfnormal(x, si^2*I)
PiError = InverseGamma(Alpha,Beta)

xi = 1./[50., 50., 50., 50.]

#######################################################

# Bookkeeping
try # save cp of this file as documentation
    cp(@__FILE__(), joinpath(simname,"$simname.jl"); remove_destination=true)
end

open(joinpath(simname,"truth.txt"), "w") do f
    println(f, "beta gamma eps s sigma eta") 
    println(f, join(round([θtrue ; σtrue],3)," ")) 
end

open(joinpath(simname,"params.txt"), "w") do f
    println(f, "n beta gamma eps s sigma eta") 
end

# initialize
bacc = 0
siacc = 0

mc = mcstart(vcat([BB[i][1:end-1] for i in 1:n]...).yy )
mcparams = mcstart([θ ; σ])




perf = @timed while true
    P = FitzHughNagumo(param(θ, σ)...)
    iter += 1


    for i in 1:n-1
        P° = MyProp(Yobs[i], Yobs[i+1], P, proptype)
        B = eulerb!(SamplePath(tts[i], yy), sample!(SamplePath(tts[i],ww), Wiener{Vec{2,Float64}}()),P°)
        if iter == 1
             BB[i] = B
        end
        llold = llikelihood(BB[i], P°)
        llnew = llikelihood(B, P°)
        
        if rand() < exp(llnew - llold) 
            bacc += 1
            BB[i] = copy(B)
        end
    end

 
    # update theta

    if iter % 1 == 0
        BBall = vcat([BB[i][1:end-1] for i in 1:n]...)
        θ° = θ + (2rand(length(θ)) .- 1).*scaleθ
        Pθ = FitzHughNagumo(param(θ, σ)...)
        Pθ° = FitzHughNagumo(param(θ°, σ)...)
        ll = girsanov(BBall, Pθ°, Pθ)
        if rand() < exp(ll)  
            θ = θ°
        end
    end


    # update conjugate theta

    θ[conjθs] = conjugateb(BB, θ[conjθs],xi[conjθs], P, phi, intercept) 
    
    
    # update sigma (and theta)
    if iter % 1 == 0
        σ° = σ .* exp(scaleσ .* randn(length(σ))) 
        θ° = θ #+ (2rand(length(θ)) .- 1).*scaleθ/3
        Pσ = FitzHughNagumo(param(θ, σ)...)
        Pσ° = FitzHughNagumo(param(θ°, σ°)...)
        ll = 0.
        for i in 1:n # reparametrization
            P° = MyProp(Yobs[i], Yobs[i+1], Pσ, proptype)
            P°° = MyProp(Yobs[i], Yobs[i+1], Pσ°, proptype)
            Z = innovations(BB[i], P°)
            BBnew[i] = eulerb(Z, P°°)
            ll += lptilde(P°°) - lptilde(P°) + llikelihood(BBnew[i], P°°) - llikelihood(BB[i], P°)
      
        end
      #  println(ll)
        if rand() < exp(ll) * (piσ²(σ°.^2)/piσ²(σ.^2)) * (prod(σ°.^2)/prod(σ.^2))
            σ = σ°
            θ = θ°
            siacc += 1
            for i in 1:n
                BB[i] = BBnew[i]
            end
         #  print("acc")
        end                    
    end     
    open(joinpath(simname,"params.txt"), "a") do f; println(f, iter, " ", join(round([θ ; σ],8)," ")) end
    println(iter, "\t", join(round([θ./θtrue; σ./σtrue; 100bacc/iter/n;  100siacc/iter  ],3),"\t"))
      
    BBall = vcat([BB[i][1:end-1] for i in 1:n]...)  
      
    
    mc = mcnext(mc, BBall.yy)  
    mcparams = mcnext(mcparams, [θ ; σ])
      
    if iter % 10 == 0
        xr = (5,7)
        
        
        plot2(Yobs, "o", "o"; xrange=xr, yrange=(-3,3),linewidth=0.1) 
        display(oplot2(BBall, "b", "g"; xrange=xr, yrange=(-3,3), linewidth=0.7))
    end
    if iter >= K break end
end

open(joinpath(simname,"info.txt"), "w") do f
    println(f, "n $n m $m T $TT A $Alpha B $Beta") 
    println(f, "Y0 = $uu") 
    println(f, "xi = $xi") 
    println(f, "acc = ", 100bacc/iter/n, " (br) ",  100siacc/iter, " (si)") 
    println(f, perf)
end



plot2(Y, "r","b" ; yrange=(-3,3),linewidth=0.5)
oplot2(Yobs,"+r", "+b")
savefig(joinpath(simname,"truth.pdf"))

plot2(Y, "r","b" ; yrange=(-3,3),linewidth=0.5)
oplot2(Yobs,"or","ob";symbolsize=0.3)
mcb = mcband(mc);
oplot2(SamplePath(tt, mcb[1]),"r","b";linewidth=0.5)
oplot2(SamplePath(tt, mcb[2]),"r","b";linewidth=0.5)
hcat(mcbandste(mcparams)..., [θtrue; σtrue])

savefig(joinpath(simname,"band.pdf"))

xr = (6,10)
plot2(Y, "r","b" ; xrange=xr,yrange=(-3,3),linewidth=0.5)
oplot2(Yobs,"or","ob";xrange=xr,symbolsize=0.3)
mcb = mcband(mc);
oplot2(SamplePath(tt, mcb[1]),"r","b";xrange=xr,linewidth=0.5)
oplot2(SamplePath(tt, mcb[2]),"r","b";xrange=xr,linewidth=0.5)
savefig(joinpath(simname,"bandpart.pdf"))



plot2(vcat([BB[i] for i in 1:n]...), "r", "b"; yrange=(-3,3), linewidth=0.5)
oplot2(Yobs,"+r", "+b")
savefig(joinpath(simname,"sample.pdf"))
