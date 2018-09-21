using Bridge, StaticArrays, Distributions
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models
using DelimitedFiles
using RCall
using DataFrames


T = 2.0
dt = 1/200
tt = 0.:dt:T

# specify target process
struct IntegratedDiffusion <: ContinuousTimeProcess{ℝ{2}}
    γ::Float64
end

Bridge.b(t, x, P::IntegratedDiffusion) = ℝ{2}(x[2], -(x[2]+sin(x[2]*x[1]) +1/2))
Bridge.σ(t, x, P::IntegratedDiffusion) = ℝ{2}(0.0, P.γ*(1+0.5*atan(x[1])))
Bridge.constdiff(::IntegratedDiffusion) = false

# Generate Data
Random.seed!(1)
P = IntegratedDiffusion(0.7)
W = sample(tt, Wiener())
x0 = ℝ{2}(2.0, 0.0)
X = solve(Euler(), x0, W, P)

# Specify observation scheme
L = @SMatrix [1. 0.]
Σ = @SMatrix [0.0]
v = ℝ{1}(16.0)    
#v = ℝ{1}(X.yy[end][1] + √Σ[1,1]*randn())
#v = ℝ{2}(X.yy[end])

# L = @SMatrix [1. 0.; 0. 1.]
# Σ = @SMatrix [0.0 0.0 ; 0.0 0.0]
# v = ℝ{2}(-5.0, 2)    #ℝ{1}(X.yy[end][1] + √Σ[1,1]*randn())

# specify auxiliary process
struct IntegratedDiffusionAux <: ContinuousTimeProcess{ℝ{2}}
    γ::Float64
end

Bridge.B(t, P::IntegratedDiffusionAux) = @SMatrix [0.0 1.0; 0.0 -1.0]
Bridge.β(t, P::IntegratedDiffusionAux) = ℝ{2}(0, 1/2)
Bridge.σ(t, x, P::IntegratedDiffusionAux) = ℝ{2}(0.0, P.γ)
Bridge.constdiff(::IntegratedDiffusionAux) = false

Bridge.b(t, x, P::IntegratedDiffusionAux) = Bridge.B(t,P) * x + Bridge.β(t,P)
Bridge.a(t, P::IntegratedDiffusionAux) = Bridge.σ(t,0,P) * Bridge.σ(t, 0, P)'

#Pt = IntegratedDiffusionAux(0.7)
Pt = IntegratedDiffusionAux(Bridge.σ(T, [v 0], P)[2])

# Solve Backward Recursion
Po = Bridge.PartialBridge(tt, P, Pt, L, v, Σ)

# initalisation
W = sample(tt, Wiener())
x0 = ℝ{2}(2.0, 1.0)
Xo = copy(X)
bridge!(Xo, x0, W, Po)
sample!(W, Wiener())
bridge!(X, x0, W, Po)
ll = llikelihood(Bridge.LeftRule(), X, Po)

# settings sampler
iterations = 50000
subsamples = 0:100:iterations
ρ = 0.9

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
    print("ll $ll $llo ")#, X[10], " ", Xo[10])
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

# write mcmc iterates to csv file 
f = open("output/integrated_diff/iterates.csv","w")
head = "iteration, time, component, value \n"
write(f, head)
iterates = [Any[s, tt[j], d, XX[i].yy[j][d]] for d in 1:2, j in 1:length(X), (i,s) in enumerate(subsamples) ][:]
writecsv(f,iterates)
close(f)

println("Average acceptance percentage: ",100*round(acc/iterations,2),"\n")

# Plotting with ggplot 
dd = DataFrame(iteration=map(x->x[1],iterates), time=map(x->x[2],iterates),
        component=map(x->x[3],iterates),value=map(x->x[4],iterates))

ind =  map(x-> x in [0:200:1000;  40000:2000:50000],dd[:iteration])
ddsub = dd[ind,:]

ggplot(ddsub,aes(x=:time,y=:value,group=:iteration,colour=:iteration)) + 
        geom_path()+facet_wrap(R"~component",ncol=1,scales="free_y")+
         scale_colour_gradient(low="green",high="blue")
    

# Calling R, somehow the pdf device does not work         
@rlibrary ggplot2
@rlibrary tidyverse
@rlibrary grDevices

R"""
library(ggplot2)
library(tidyverse)
library(grDevices)

theme_set(theme_minimal())
setwd("output/integrated_diff")

d <- read.csv("output/integrated_diff/iterates.csv")
dsub <- d[d$iteration %in% c(seq(0,1000,by=200),seq(40000,50000,by=2000)),]


pdf("iterates.pdf",width=8,height=7)
dsub %>% ggplot(aes(x=time,y=value,colour=iteration)) +
  geom_path(aes(group=iteration)) +
  facet_wrap(~component,ncol=1,scales='free_y')+
  scale_colour_gradient(low='yellow',high='Darkred')
dev.off()    
"""

# import variables in Julia 
@rget d


