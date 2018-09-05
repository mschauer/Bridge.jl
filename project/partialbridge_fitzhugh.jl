cd("/Users/Frank/.julia/dev/Bridge")

using Bridge, StaticArrays, Distributions
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models
using DelimitedFiles
using RCall
using DataFrames
using CSV

@rlibrary ggplot2

T = 2.0
dt = 1/5000
tt = 0.:dt:T

aux_choice = ["linearised_end" "linearised_startend" "matching"][1]
endpoint = ["first", "extreme"][2]


# specify target process
struct FitzhughDiffusion <: ContinuousTimeProcess{ℝ{2}}
    ϵ::Float64
    s::Float64
    γ::Float64
    β::Float64
    σ::Float64      
end


#Bridge.B(t, P::Union{FitzhughDiffusion,FitzhughDiffusionAux}) = @SMatrix [-1.0 1.0; 0.0 -1.0]

Bridge.b(t, x, P::FitzhughDiffusion) = ℝ{2}((x[1]-x[2]-x[1]^3+P.s)/P.ϵ, P.γ*x[1]-x[2] +P.β)
Bridge.σ(t, x, P::FitzhughDiffusion) = ℝ{2}(0.0, P.σ)
Bridge.constdiff(::FitzhughDiffusion) = true

# Generate Data
Random.seed!(3)
P = FitzhughDiffusion(0.1, 0.0, 1.5, 0.8, 0.3)
W = sample(tt, Wiener())
#x0 = ℝ{2}(-0.5, 0.0) # is a good choice, with either v=-0.5 or where the forward process ends
x0 = ℝ{2}(-0.5, -0.6)
X = solve(Euler(), x0, W, P)    

XX = [X]
samples = 250
#draw 9 more path and save these
for j in 2:samples
    W = sample(tt, Wiener())
    X = solve(Euler(), x0, W, P)
    push!(XX, X)
end

# write forwards to csv file 
f = open("/Users/Frank/Dropbox/DiffBridges/Rcode/integrated_diff/forwards.csv","w")
head = "iteration, time, component, value \n"
write(f, head)
iterates = [Any[i, tt[j], d, XX[i].yy[j][d]] for d in 1:2, j in 1:length(X), i in 1:samples ][:]
writedlm(f, iterates, ',')
close(f)         

# Choose truepath
if endpoint == "first"
    itrue = 1
elseif endpoint == "extreme"
    itrue = findmax([XX[i].yy[end][1] for i in 1:samples])[2]    
else
    error("not implemented")
end


lt = size(X.tt)[1]
trueX = DataFrame(iteration=fill(-50000,2*lt),time=[X.tt; X.tt], 
    component=[fill(1,lt);fill(2,lt)], value=[first.(XX[itrue].yy); last.(XX[itrue].yy)] )
ggplot(trueX,aes(x=:time,y=:value)) + 
      geom_path()+facet_wrap(R"~component",ncol=1,scales="free_y")



# Specify observation scheme
L = @SMatrix [1. 0.]
Σ = @SMatrix [0.0]
#v = ℝ{1}(-0.5)    

v = ℝ{1}(XX[itrue].yy[end][1] + √Σ[1,1]*randn())
#v = ℝ{2}(X.yy[end])

# L = @SMatrix [1. 0.; 0. 1.]
# Σ = @SMatrix [0.0 0.0 ; 0.0 0.0]
# v = ℝ{2}(-5.0, 2)    #ℝ{1}(X.yy[end][1] + √Σ[1,1]*randn())

# specify auxiliary process
struct FitzhughDiffusionAux <: ContinuousTimeProcess{ℝ{2}}
    ϵ::Float64
    s::Float64
    γ::Float64
    β::Float64
    σ::Float64
    t::Float64
    u::Float64
    T::Float64
    v::Float64      
end

function uv(t, P::FitzhughDiffusionAux) 
    λ = (t - P.t)/(P.T - P.t)
    P.v*λ + P.u*(1-λ)
end

if aux_choice=="linearised_end"
    Bridge.B(t, P::FitzhughDiffusionAux) = @SMatrix [1/P.ϵ-3*P.v^2/P.ϵ  -1/P.ϵ; P.γ -1.0]
    Bridge.β(t, P::FitzhughDiffusionAux) = ℝ{2}(P.s/P.ϵ+2*P.v^3/P.ϵ, P.β)
    ρ = endpoint=="extreme" ? 0.99 : 0.6
elseif aux_choice=="linearised_startend"
    Bridge.B(t, P::FitzhughDiffusionAux) = @SMatrix [1/P.ϵ-3*uv(t, P)^2/P.ϵ  -1/P.ϵ; P.γ -1.0]
    Bridge.β(t, P::FitzhughDiffusionAux) = ℝ{2}(P.s/P.ϵ+2*uv(t, P)^3/P.ϵ, P.β)
    ρ = endpoint=="extreme" ? 0.99 : 0.6
else
    Bridge.B(t, P::FitzhughDiffusionAux) = @SMatrix [1/P.ϵ  -1/P.ϵ; P.γ -1.0]
    Bridge.β(t, P::FitzhughDiffusionAux) = ℝ{2}(P.s/P.ϵ-(P.v^3)/P.ϵ, P.β)
    ρ = 0.99
end

Bridge.σ(t, x, P::FitzhughDiffusionAux) = ℝ{2}(0.0, P.σ)
Bridge.constdiff(::FitzhughDiffusionAux) = true

Bridge.b(t, x, P::FitzhughDiffusionAux) = Bridge.B(t,P) * x + Bridge.β(t,P)
Bridge.a(t, P::FitzhughDiffusionAux) = Bridge.σ(t,0,P) * Bridge.σ(t, 0, P)'

#Pt = IntegratedDiffusionAux(0.7)
Pt = FitzhughDiffusionAux(0.1, 0.0, 1.5, 0.8, 0.3, tt[1], x0[1], tt[end], v[1])

# Solve Backward Recursion
Po = Bridge.PartialBridge(tt, P, Pt, L, v, Σ)

# initalisation
W = sample(tt, Wiener())
Xo = copy(X)
bridge!(Xo, x0, W, Po)
sample!(W, Wiener())
bridge!(X, x0, W, Po)
ll = llikelihood(Bridge.LeftRule(), X, Po)

# settings sampler
iterations = 10*10^4
subsamples = 0:1000:iterations


# further initialisation
Wo = copy(W)
W2 = copy(W)
XX = Any[]
if 0 in subsamples
    push!(XX, copy(X))
end

acc = 0 

for iter in 1:iterations
    # Proposal
    sample!(W2, Wiener())
    Wo.yy .= ρ*W.yy + sqrt(1-ρ^2)*W2.yy
    bridge!(Xo, x0, Wo, Po)

    llo = llikelihood(Bridge.LeftRule(), Xo, Po)
    print("ll $ll $llo, diff_ll: ",round(llo-ll,3))#, X[10], " ", Xo[10])
    
    if log(rand()) <= llo - ll
        X.yy .= Xo.yy
        W.yy .= Wo.yy
        ll = llo
        print("✓")
        acc +=1 
    end    
    println()
    if iter in subsamples
        push!(XX, copy(X))
    end
end

@info "Done."*"\x7"^6

# write mcmc iterates to csv file 
f = open("/Users/Frank/Dropbox/DiffBridges/Rcode/integrated_diff/iterates.csv","w")
head = "iteration, time, component, value \n"
write(f, head)
iterates = [Any[s, tt[j], d, XX[i].yy[j][d]] for d in 1:2, j in 1:length(X), (i,s) in enumerate(subsamples) ][:]
writecsv(f,iterates)
close(f)

# write a true path to csv file 
CSV.write("/Users/Frank/Dropbox/DiffBridges/Rcode/integrated_diff/truepath.csv",trueX)



println("Average acceptance percentage: ",100*round(acc/iterations,2),"\n")

# Plotting with ggplot 
dd = DataFrame(iteration=map(x->x[1],iterates), time=map(x->x[2],iterates),
        component=map(x->x[3],iterates),value=map(x->x[4],iterates))

ind =  map(x-> x in [0:200:1000;  40000:2000:50000],dd[:iteration])
ddsub = dd[ind,:]

ggplot([trueX;ddsub],aes(x=:time,y=:value,group=:iteration,colour=:iteration)) + 
        geom_path()+facet_wrap(R"~component",ncol=1,scales="free_y")+
         scale_colour_gradient(low="green",high="blue")            |>
 p -> ggsave("/Users/Frank/Dropbox/DiffBridges/Rcode/integrated_diff/p1.pdf", p, width=7,height=4)
    

ind2 =  map(x-> x in 50000:2000:100000,dd[:iteration])
ddsub2 = dd[ind2,:]

ggplot([trueX;ddsub2],aes(x=:time,y=:value,group=:iteration,colour=:iteration)) + 
        geom_path()+facet_wrap(R"~component",ncol=1,scales="free_y")+
         scale_colour_gradient(low="green",high="blue")            |>
 p -> ggsave("/Users/Frank/Dropbox/DiffBridges/Rcode/integrated_diff/p2.pdf", p, width=7,height=4)
