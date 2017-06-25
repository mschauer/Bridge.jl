using Bridge, Distributions

PLOT = false
include("plot.jl")

struct Atan <: ContinuousTimeProcess{Float64}
    α::Float64
    β::Float64 
    σ::Float64
    a::Float64
    Atan(a,b,c) = new(a,b,c,c^2)
end


function param(θ, σ)
        θ[1], θ[2], σ[1] 
end
 
Bridge.b(t, x, P::Atan) =  P.α*atan(x) + P.β 
phi12(t, x, P::Atan) = [atan(x), 1.]
intercept12(t, x, P::Atan) = 0.

Bridge.σ(t, x, P::Atan) = P.σ
Bridge.a(t, x, P::Atan) = P.a
Bridge.Γ(t, x, P::Atan) = inv(P.a)


function conjugateb(YY, th, xi, P, phif, intc)
        n = length(th)
        G = zeros(n,n)
        mu = zeros(th)
        for Y in YY
            for i in 1:length(Y)-1
                phi = phif(Y[i]..., P)
                Gphi = Bridge.Γ(Y[i]..., P)*phi
                zi = phi*Gphi'
                dy = Y.yy[i+1] - Y.yy[i] 
                ds = Y.tt[i+1] - Y.tt[i] 
                mu = mu + Gphi*(dy - intc(Y[i]..., P)*ds)
                G[:] = G +  zi * (Y.tt[i+1]-Y.tt[i])
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

function MyProp(u, v, P, proptype=:mbb)
    if proptype == :guip
        #Pt = LinPro(P.α*cos(-P.β/P.α)^2, 0.5*P.α*sin(2P.β/P.α), P.σ)
        Pt = LinPro(P.α*cos(-P.β/P.α)^2, tan(-P.β/P.α), P.σ)
        return GuidedProp(P, u..., v..., Pt)
    else
        return DHBridgeProp(P, u..., v...)
    end
end



############## Configuration ###################################
srand(10)
K = 100
m = 10 # number of euler steps per segments

simid = 2
propid = 2
proptype = [:mbb,:guip][propid]

simname =["atan", "atantc"][simid]
# simname =["atan", "atantc"][simid] * "$proptype$m"


θ = [-2., 0.]  
θtrue=copy(θ)
scaleθ = 0.08*[1., 1.]
σ = [0.75]
scaleσ = [0.1]
σtrue = copy(σ)
Ptrue = Atan(param(θ, σ)...)
 


n = 100 # number of segments
mextra = 2*div(1000,m) #factor of extra steps for truth
TT = 30.
dt = TT/n/m
tt = linspace(0., TT, n*m+1)
tttrue = linspace(0., TT, n*mextra*m+1)
ttf = tt[1:m:end]

uu = 0.

r = [(:xrange,(-2,2)), (:yrange,(-1,3))]

#######################################################

Y = euler(uu, sample(tttrue, Wiener{Float64}()), Ptrue) 
Yobs = Y[1:mextra*m:end] #subsample
assert(endof(Yobs) == n+1)


Y2 = SamplePath(tt, copy(Y.yy[1:mextra:end])) #subsample 
normalize(tt) = (tt - tt[1])/(tt[end]-tt[1])

θs = Float64[]
iter = 0
tts = [collect(tt[m*(i-1)+1:m*i+1]) for i in 1:n]
if simid == 2
    tts = [Bridge.tau(tts[i]) for i in 1:n]
end

BB = SamplePath[Y2[m*(i-1)+1:m*i+1] for i in 1:n]
for i in 1:n
    BB[i].tt[:] = tts[i]
    BB[i].yy[:] =  map(x -> BB[i].yy[1] + x*(BB[i].yy[end] - BB[i].yy[1]),normalize(tts[i]))
end
    
BBnew = copy(BB)
BBall = vcat([BB[i][1:end-1] for i in 1:n]...)

# 
if PLOT
        xr = (5,7)
        plot(Yobs, "o"; xrange=xr, yrange=(-3,3),linewidth=0.1) 
        display(oplot(BBall,"b"; xrange=xr, yrange=(-3,3), linewidth=0.7))
end

################### Prior ###################################
conjθs = [1,2] #set of conjugate thetas
phi = phi12 
intercept = intercept12 
estθ = [true, true] #params to estimate

# arbitrary starting values
 
σ = [2.]
θ = [-0.1, -0.1]

for i in 1:length(θ) # start with truth for parameters not to be estimated
    if !estθ[i]
        θ[i] = θtrue[i]
        scaleθ[i] = 0.
    end
end    

P = Atan(param(θ, σ)...)
 
    


# reserve some space
i = 1
ww = Array{Float64,1}(length(m*(i-1)+1:m*(i)+1)) 
yy = copy(ww)

# Prior
Alpha = 1/500
Beta= 1/500

piσ²(s2) = pdf(InverseGamma(Alpha,Beta), s2[1])
lq(x, si) = Bridge.logpdfnormal(x, si^2)
PiError = InverseGamma(Alpha,Beta)

xi = 1./[5., 5.] #prior prec

#######################################################

# Bookkeeping
mkpath(joinpath("output",simname))
try # save cp of this file as documentation
    cp(@__FILE__(), joinpath("output",simname,"$simname.jl"); remove_destination=true)
end

open(joinpath("output",simname,"truth.txt"), "w") do f
    println(f, "alpha beta sigma") 
    println(f, join(round.([θtrue ; σtrue],3)," ")) 
end

open(joinpath("output",simname,"params.txt"), "w") do f
    println(f, "n alpha beta sigma") 
end

# initialize
bacc = 0
siacc = 0

mc = mcstart(vcat([BB[i][1:end-1] for i in 1:n]...).yy )
mcparams = mcstart([θ ; σ])

Bridge.constdiff(::Atan) = true

perf = @timed while true
    P = Atan(param(θ, σ)...)
    iter += 1


    for i in 1:n-1
        P° = MyProp(Yobs[i], Yobs[i+1], P, proptype)
        B = bridge!(SamplePath(tts[i], yy), sample!(SamplePath(tts[i],ww), Wiener{Float64}()),P°)
        # B = shiftedeulerb!(SamplePath(tts[i], yy), sample!(SamplePath(tts[i],ww), Wiener{Float64}()),P°)
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
        #BBall = vcat([BB[i][1:end-1] for i in 1:n]...)
        θ° = θ + (2rand(length(θ)) .- 1).*scaleθ
        Pθ = Atan(param(θ, σ)...)
        Pθ° = Atan(param(θ°, σ)...)
        #ll = girsanov(BBall, Pθ°, Pθ)
        ll = 0.
        for i in 1:n
            ll += girsanov(BB[i], Pθ°, Pθ)
        end    
        if rand() < exp(ll)  
            θ = θ°
        end
    end


    # update conjugate theta

    θ[conjθs] = conjugateb(BB, θ[conjθs],xi[conjθs], P, phi, intercept) 
    
    
    # update sigma (and theta)
    if iter % 1 == 0
        σ° = σ .* exp.(scaleσ .* randn(length(σ))) 
        θ° = θ #+ (2rand(length(θ)) .- 1).*scaleθ/3
        Pσ = Atan(param(θ, σ)...)
        Pσ° = Atan(param(θ°, σ°)...)
        ll = 0.
        for i in 1:n # reparametrization
            P° = MyProp(Yobs[i], Yobs[i+1], Pσ, proptype)
            P°° = MyProp(Yobs[i], Yobs[i+1], Pσ°, proptype)
            Z = innovations(BB[i], P°)
            BBnew[i] = bridge(Z, P°°)
            # BBnew[i] = eulerb(Z, P°°)
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
    open(joinpath("output",simname,"params.txt"), "a") do f; println(f, iter, " ", join(round.([θ ; σ],8)," ")) end
    println(iter, "\t", join(round.([θ; σ./σtrue; 100bacc/iter/n;  100siacc/iter  ],3),"\t"))
      
    BBall = vcat([BB[i][1:end-1] for i in 1:n]...)  
      
    
    mc = mcnext(mc, BBall.yy)  
    mcparams = mcnext(mcparams, [θ ; σ])
      
    if iter % 10 == 0
        xr = (5,7)
        
        if PLOT
            plot(Yobs, "o", "o"; xrange=xr, yrange=(-3,3),linewidth=0.1) 
            display(oplot(BBall, "b", "g"; xrange=xr, yrange=(-3,3), linewidth=0.7))
        end    
    end
    if iter >= K break end
end

open(joinpath("output",simname,"info.txt"), "w") do f
    println(f, "n $n m $m T $TT A $Alpha B $Beta") 
    println(f, "Y0 = $uu") 
    println(f, "xi = $xi") 
    println(f, "acc = ", 100bacc/iter/n, " (br) ",  100siacc/iter, " (si)") 
    println(f, perf)
end

if PLOT

plot(Y, "r","b" ; yrange=(-3,3),linewidth=0.5)
oplot(Yobs,"+r", "+b")
savefig(joinpath("output",simname,"truth.pdf"))

plot(Y, "r","b" ; yrange=(-3,3),linewidth=0.5)
oplot(Yobs,"or","ob";symbolsize=0.3)
mcb = mcband(mc);
oplot(SamplePath(tt, mcb[1]),"r","b";linewidth=0.5)
oplot(SamplePath(tt, mcb[2]),"r","b";linewidth=0.5)
hcat(mcbandste(mcparams)..., [θtrue; σtrue])

savefig(joinpath("output",simname,"band.pdf"))

xr = (6,10)
plot(Y, "r","b" ; xrange=xr,yrange=(-3,3),linewidth=0.5)
oplot(Yobs,"or","ob";xrange=xr,symbolsize=0.3)
mcb = mcband(mc);
oplot(SamplePath(tt, mcb[1]),"r","b";xrange=xr,linewidth=0.5)
oplot(SamplePath(tt, mcb[2]),"r","b";xrange=xr,linewidth=0.5)
savefig(joinpath("output",simname,"bandpart.pdf"))



plot(vcat([BB[i] for i in 1:n]...), "r", "b"; yrange=(-3,3), linewidth=0.5)
oplot(Yobs,"+r", "+b")
savefig(joinpath("output",simname,"sample.pdf"))
end 