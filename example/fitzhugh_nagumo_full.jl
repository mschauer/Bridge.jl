using Bridge, Distributions, StaticArrays
PLOT = :pyplot
include("plot.jl")

#diag2(x, y) = SDiagonal(@SVector [x,y])
diag2(x, y) = @SMatrix [x 0. ; 0. y]

# 2: 1655 seconds

struct FitzHughNagumo  <: ContinuousTimeProcess{SVector{2,Float64}}
    α::Float64
    β::Float64 
    γ1::Float64
    γ2::Float64
    ϵ::Float64
    s::Float64
#    σ::SDiagonal{2,Float64}
#    a::SDiagonal{2,Float64}
    σ::SMatrix{2,2,Float64}
    a::SMatrix{2,2,Float64}

    FitzHughNagumo(σ, α = 1/3,  β = 0.08*0.7, γ1 = 0.08, γ2 = 0.08*0.8,  ϵ = 1.,  s = 1.) = new(α, β, γ1, γ2, ϵ, s, σ, σ*σ') #'
end


#Bridge.b(t, x, P::FitzHughNagumo) = @SVector [(-P.α*x[1]^3+ P.ϵ*(x[1]-x[2]) + P.s, P.γ1*x[1]- P.γ2*x[2] + P.β)]
#phi234(t, x, P::FitzHughNagumo) = @SMatrix [0. -x[1]^3 + x[1]-x[2] 1. ; x[1] 0. 0.] #γ, ϵ, s 
function param(θ, σ)
    β, γ, ϵ, s = θ
    σ1, σ2 = σ
#   σ,    α, β, γ1,γ2, ϵ,  s 
    diag2(σ1,σ2), ϵ, β, γ, 1., ϵ, s  # α == ϵ
end
 
Bridge.b(t, x, P::FitzHughNagumo) = @SVector [-P.α*x[1]^3 + P.ϵ*(x[1]-x[2]) + P.s, P.γ1*x[1]- P.γ2*x[2] + P.β]
phi1234(t, x, P::FitzHughNagumo) = @SMatrix [0. 0. -x[1]^3 + x[1]-x[2] 1. ; 1. x[1] 0. 0.] #β, γ, ϵ, s
intercept1234(t, x, P::FitzHughNagumo) = @SVector [0, - P.γ2*x[2]]

Bridge.σ(t, x, P::FitzHughNagumo) = P.σ
Bridge.a(t, x, P::FitzHughNagumo) = P.a
Bridge.Γ(t, x, P::FitzHughNagumo) = inv(P.a)
Bridge.constdiff(::FitzHughNagumo) = true

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
    m2 = m2 + map((x,y)->x.*y, delta, x - m)
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

function MyProp(u, v, P, proptype=:mdb)
    if proptype == :aff
        cs = Bridge.CSpline(u[1], v[1],  Bridge.b(u..., P),   Bridge.b(v..., P))
        #println(Bridge.b(u..., P),  Bridge.b(v..., P))
        return BridgeProp(P, u..., v..., P.a, cs)
    elseif proptype == :lin
        error("proptype $proptype not implemented")
#        y = -3*a^2*P.α
#        B = ...
#        β = ...
#        Pt = Bridge.LinPro(B, -B\β, P.σ)
        return GuidedProp(P, u..., v..., Pt)
    elseif proptype == :dh
        return DHBridgeProp(P, u..., v...)
    else        
        error("proptype $proptype not implemented")
    end
end


function main(propid, m, budget)

############## Configuration ###################################
srand(10)
K = 100000 #100000
tim = time()
budget *= 60

simid = 1
#propid = 1
proptype = [:dh, :aff, :aff][propid]
eulertype = [:mdb, :eul, :tcs][propid]
simname =["full", "fullne"][simid] * "$proptype$eulertype$m"
println(simname)

STIME = false

θ =  [[0.6, 1.4][simid], 1.5, 10., 0.5*10] # [β, γ, ϵ, s] 
θtrue=copy(θ)
scaleθ = 0.04*[1., 1., 5., 5.]
σ = [0.25, 0.2]
#σ = [0.20, 0.15]
scaleσ = [0.02,0.02] #([0.01, 0.01],[0.01, 0.01],[0.01, 0.01])[propid]

σtrue = copy(σ)
Ptrue = FitzHughNagumo(param(θtrue, σtrue)...)
 


#n = 500 # number of segments
n = 3
#m = 80 # number of euler steps per segments
mextra = div(500, m) #factor of extra steps for truth
#TT = 150.
TT = 1.
dt = TT/n/m
println("dt $dt")
tt = linspace(0., TT, n*m+1)
tttrue = linspace(0., TT, n*mextra*m+1)
ttf = tt[1:m:end]

uu = @SVector [0., 1.]

r = [(:xrange,(-2,2)), (:yrange,(-1,3))]

#######################################################

global Y = euler(uu, sample(tttrue, Wiener{SVector{2,Float64}}()), Ptrue) 
global Yobs = Y[1:mextra*m:end] #subsample
assert(endof(Yobs) == n+1)


Y2 = SamplePath(tt, copy(Y.yy[1:mextra:end])) #subsample 
normalize(tt) = (tt - tt[1])/(tt[end]-tt[1])
tau(t, tmin, tmax) = tmin + (t-tmin).*(2-(t-tmin)/(tmax-tmin))

θs = Float64[]
iter = 0
if STIME
    tts = [tau(collect(tt[m*(i-1)+1:m*(i)+1]), tt[m*(i-1)+1],tt[m*(i)+1]) for i in 1:n]
else
    tts = [(collect(tt[m*(i-1)+1:m*(i)+1])) for i in 1:n]
end
sss = deepcopy(tts)

BB = SamplePath[Y2[m*(i-1)+1:m*i+1] for i in 1:n]
for i in 1:n
    BB[i].yy[:] =  map(x -> BB[i].yy[1] + x*(BB[i].yy[end] - BB[i].yy[1]),normalize(tts[i]))
end
    
BBnew = copy(BB)
BBall = vcat([BB[i][1:end-1] for i in 1:n]...)

# 
if PLOT == :winston
        xr = (5,7)
        plot2(Yobs, "o", "o"; xrange=xr, yrange=(-3,3),linewidth=0.1) 
        display(oplot2(BBall, "b", "b"; xrange=xr, yrange=(-3,3), linewidth=0.7))
elseif PLOT == :pyplot
#    PyPlot.clf();plot(Y[1:3000], linewidth=0.2, color="b"); plot(Yobs[1:5], linewidth=0, marker="o")
end



################### Prior ###################################
conjθs = [1,2,3,4] #set of conjugate thetas
phi = phi1234
intercept = intercept1234
#estθ = [false, false, true, false] #params to estimate
estθ = [true, false, false, false] #params to estimate
estσ = [false, true]
# arbitrary starting values
 
σ[2] = 0.3
#σ = copy(σtrue)
θ = 0.5 + 1.0*rand(length(θ)) 
#θ = copy(θtrue)

for i in 1:length(θ) # start with truth for parameters not to be estimated
    if !estθ[i]
        θ[i] = θtrue[i]
        scaleθ[i] = 0.
    end
end  
for i in 1:length(σ) # start with truth for parameters not to be estimated
    if !estσ[i]
        σ[i] = σtrue[i]
        scaleσ[i] = 0.
    end
end    


P = FitzHughNagumo(param(θ, σ)...)

#if PLOT
#    plot2(Y, "r","r" ; yrange=(-3,3),linewidth=0.5)
#    display(oplot2(Y2, "b", "b"; yrange=(-3,3), linewidth=0.7))
#end
    


# reserve some space
i = 1
ww = Array{SVector{2,Float64},1}(length(m*(i-1)+1:m*(i)+1)) ## checkme!
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
mkpath(joinpath("output",simname))
try # save cp of this file as documentation
    cp(@__FILE__(), joinpath("output",simname,"$simname.jl"); remove_destination=true)
end

open(joinpath("output",simname,"truth.txt"), "w") do f
    println(f, "beta gamma eps s sigma eta") 
    println(f, join(round.([θtrue ; σtrue],3)," ")) 
end

open(joinpath("output",simname,"params.txt"), "w") do f
    println(f, "n beta gamma eps s sigma eta") 
end

# initialize
bacc = zeros(Int, n-1)
siacc = 0
thacc = 0

mc = mcstart(vcat([BB[i][1:end-1] for i in 1:n]...).yy )
mcparams = mcstart([θ ; σ])




perf = @timed while true

    #σ[:] = σtrue
    #θ[:] = θtrue

    P = FitzHughNagumo(param(θ, σ)...)
    iter += 1


    for i in 1:n-1
        P° = MyProp(Yobs[i], Yobs[i+1], P, proptype)
        if eulertype == :tcs
            B = ubridge!(SamplePath(sss[i], yy), sample!(SamplePath(tts[i],ww), Wiener{SVector{2,Float64}}()),P°)
        elseif eulertype == :mdb
            B = bridge!(SamplePath(sss[i], yy), sample!(SamplePath(tts[i],ww), Wiener{SVector{2,Float64}}()),P°, Bridge.mdb!)
        elseif eulertype == :eul
            B = bridge!(SamplePath(sss[i], yy), sample!(SamplePath(tts[i],ww), Wiener{SVector{2,Float64}}()),P°)
        else        
            error("$eulertype not implemented")
        end
        if iter == 1
             BB[i] = B
        end
        if eulertype == :tcs
            llold = llikelihood(BB[i], P°)
            llnew = llikelihood(B, P°)
        else
            llold = llikelihood(BB[i], P°)
            llnew = llikelihood(B, P°)
        end
        if rand() < exp(llnew - llold) 
            bacc[i] += 1
            BB[i] = copy(B)
        end
    end

 
    # update theta

    if iter % 1 == 1
        BBall = vcat([BB[i][1:end-1] for i in 1:n]...)
        θ° = θ + (2rand(length(θ)) .- 1).*scaleθ
        Pθ = FitzHughNagumo(param(θ, σ)...)
        Pθ° = FitzHughNagumo(param(θ°, σ)...)
        ll = girsanov(BBall, Pθ°, Pθ)
        if rand() < exp(ll)  
            θ = θ°
            thacc += 1
        end
    end


    #update conjugate theta
    #if any(estθ[conjθs])
    #    θ[conjθs[estθ]] = conjugateb(BB, θ[conjθs],xi[conjθs], P, phi, intercept)[estθ[conjθs]]
    #end
    
    # update sigma (and theta)
    if iter % 1 == 0
        σ° = σ .* exp.(scaleσ .* randn(length(σ))) 
        θ° = θ + (2rand(length(θ)) .- 1).*scaleθ/3
        Pσ = FitzHughNagumo(param(θ, σ)...)
        Pσ° = FitzHughNagumo(param(θ°, σ°)...)
        ll = 0.
        for i in 1:n # reparametrization
            P° = MyProp(Yobs[i], Yobs[i+1], Pσ, proptype)
            P°° = MyProp(Yobs[i], Yobs[i+1], Pσ°, proptype)
            if eulertype == :mdb 
                Z = Bridge.mdbinnovations(BB[i], P°)
                #Z2 = sample(Z.tt, Wiener{SVector{2,Float64}}())
                #Z.yy[:] = sqrt(.9)*Z.yy + sqrt(0.1)*Z2.yy
                BBnew[i] = bridge(Z, P°°, Bridge.mdb!)
                ll += lptilde(P°°) - lptilde(P°) + llikelihood(BBnew[i], P°°) - llikelihood(BB[i], P°)
            elseif eulertype == :tcs
                Z = Bridge.uinnovations(BB[i], P°)
                #Z2 = sample(Z.tt, Wiener{SVector{2,Float64}}())
                #Z.yy[:] = sqrt(.9)*Z.yy + sqrt(0.1)*Z2.yy
                BBnew[i] = Bridge.ubridge(Z, P°°)
                ll += lptilde(P°°) - lptilde(P°) + ullikelihood(BBnew[i], P°°) - ullikelihood(BB[i], P°)
            elseif eulertype == :eul
                Z = innovations(BB[i], P°)
                BBnew[i] = bridge(Z, P°°)
                ll += lptilde(P°°) - lptilde(P°) + llikelihood(BBnew[i], P°°) - llikelihood(BB[i], P°)
            else        
                error("$eulertype not implemented")
            end
            
        end
        
        if rand() < exp(ll) #* (piσ²(σ°.^2)/piσ²(σ.^2)) * (prod(σ°.^2)/prod(σ.^2))
        
            σ = σ°
            θ = θ°
            siacc += 1
            for i in 1:n
                BB[i] = BBnew[i]
            end
         #  print("acc")
        end                    
    end     
    open(joinpath("output",simname,"params.txt"), "a") do f; println(f, iter, " ", join(round.([θ ; σ],8)," ")) end
    println(iter, "\t", join(round.([θ./θtrue; σ./σtrue; Inf; 100mean(bacc)/iter; 100minimum(bacc)/iter; Inf; 100thacc/iter; 100siacc/iter; ll  ],3),"\t")) 
      
    BBall = vcat([BB[i][1:end-1] for i in 1:n]...)  
      
    
    mc = mcnext(mc, BBall.yy)  
    mcparams = mcnext(mcparams, [θ ; σ])
    if PLOT == :winston && iter % 10 == 0
        xr = (5,7)
        
        
        plot2(Yobs, "o", "o"; xrange=xr, yrange=(-3,3),linewidth=0.1) 
        display(oplot2(BBall, "b", "g"; xrange=xr, yrange=(-3,3), linewidth=0.7))
    elseif PLOT == :pyplot && iter % 10 == 1
        #xr = (5,7)
                
        if iter % 1000 == 1
            PyPlot.clf()
            display(plot(Yobs[1:min(10,n)]; linewidth=0.0, marker="o", color="b"))     
        end
        
        display(plot(BBall[m*1:m*min(9, n-1)], linewidth=0.1, color="b"))    
    end
    if iter >= K || abs(time() - tim) > budget 
        break
    end
end

open(joinpath("output",simname,"info.txt"), "w") do f
    println(f, "n $n m $m T $TT A $Alpha B $Beta") 
    println(f, "Y0 = $uu") 
    println(f, "xi = $xi") 
    println(f, "acc = ", 100bacc/iter, " (br) ",  100siacc/iter, " (si)",  100thacc/iter, " (th)") 
    println(f, perf)
end

if PLOT == :winston

    plot2(Y, "r","b" ; yrange=(-3,3),linewidth=0.5)
    oplot2(Yobs,"+r", "+b")
    savefig(joinpath("output",simname,"truth.pdf"))

    plot2(Y, "r","b" ; yrange=(-3,3),linewidth=0.5)
    oplot2(Yobs,"or","ob";symbolsize=0.3)
    mcb = mcband(mc);
    oplot2(SamplePath(tt, mcb[1]),"r","b";linewidth=0.5)
    oplot2(SamplePath(tt, mcb[2]),"r","b";linewidth=0.5)
    hcat(mcbandste(mcparams)..., [θtrue; σtrue])

    savefig(joinpath("output",simname,"band.pdf"))

    xr = (6,10)
    plot2(Y, "r","b" ; xrange=xr,yrange=(-3,3),linewidth=0.5)
    oplot2(Yobs,"or","ob";xrange=xr,symbolsize=0.3)
    mcb = mcband(mc);
    oplot2(SamplePath(tt, mcb[1]),"r","b";xrange=xr,linewidth=0.5)
    oplot2(SamplePath(tt, mcb[2]),"r","b";xrange=xr,linewidth=0.5)
    savefig(joinpath("output",simname,"bandpart.pdf"))



    plot2(vcat([BB[i] for i in 1:n]...), "r", "b"; yrange=(-3,3), linewidth=0.5)
    oplot2(Yobs,"+r", "+b")
    savefig(joinpath("output",simname,"sample.pdf"))
end 
mc, mcparams
end
#include("fitzhugh_nagumo_full.jl"); for i in 1:3, m in [10,20,50]; main(i, m, 0.3); end
