# reminder, to type H*, do H\^+
#cd("/Users/Frank/.julia/dev/Bridge")
#outdir="output/landmarks/"

using Bridge, StaticArrays, Distributions
using Bridge:logpdfnormal
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models
using DelimitedFiles
using DataFrames
using CSV
using RCall
using Base.Iterators
using SparseArrays

models = [:ms, :ahs]
TEST = true

include("state.jl")
include("msmodel.jl")
include("ahsmodel.jl")
include("patches.jl")


n = 250

t = 0.0:0.001:1.0


a = 1.0 ; λ = 0.5 ; γ = 5.0
Pms = MarslandShardlow(a, γ, λ, n)

nfvar = 1.0 # tau
r1 = (-2.0:2.0) #+ 0.1rand(5)
r2 = (-2.0:2.0) #+ 0.1rand(5)

nfloc = Point.(collect(product(r1, r2)))[:]
nfscales = [0.5Point(1.0, 1.0) for x in nfloc]
nfs = [Noisefield(δ, λ, nfvar) for (δ, λ) in zip(nfloc, nfscales)]
Pahs = Landmarks(a, λ, n, nfs)

model = models[2]

if model == :ms
    dwiener = n
    StateW = Point
    P = Pms
else
    dwiener = length(nfloc)
    StateW = Float64
    P = Pahs
end

w0 = zeros(StateW, dwiener)
W = SamplePath(t, [copy(w0) for s in t])
sample!(W, Wiener{Vector{StateW}}())

q0 = [Point(cos(t), sin(t)) for t in (0:(2pi/n):2pi)[1:n]]
#q0 = circshift(q0, (1,))

p_ = 3*Point(0.1, 0.1)
p0 = [p_ for i in 1:n]
#p0 = [randn(Point) for i in 1:n]
x0 = State(q0, p0)
X = SamplePath(t, [copy(x0) for s in t])
@time solve!(EulerMaruyama!(), X, x0, W, P)

extractcomp(v,i) = map(x->x[i], v)

out = [Any[X.tt[i], [X.yy[i][CartesianIndex(c, k)][l] for l in 1:d, c in 1:2]..., "point$k"] for k in 1:n, i in eachindex(X.tt) ][:]
df = DataFrame(time=extractcomp(out,1),pos1=extractcomp(out,2),pos2=extractcomp(out,3),mom1=extractcomp(out,4),mom2=extractcomp(out,5),pointID=extractcomp(out,6))
#Any["X.tt[$i]", ["X.yy[$i][CartesianIndex($c, $k)][$l]" for l in 1:d, k in 1:n, c in 1:2]...]
@rput df
R"""
library(tidyverse)
df %>% dplyr::select(time,pos1,pos2,pointID) %>%
     ggplot(aes(x=pos1,pos2,group=pointID,colour=time)) + geom_path()
"""

R"""
df %>% dplyr::select(time,mom1,mom2,pointID) %>%
    filter(pointID %in% c("point1","point2","point3","point4","point5")) %>%
     ggplot(aes(x=mom1,mom2,group=pointID,colour=time)) + geom_path() +
     facet_wrap(~pointID)
"""


xT = X.yy[end]
Pmsaux = MarslandShardlowAux(Pms, xT)

if TEST
    include("testms.jl")
end

#= Later:

Pauxahs = LandmarksAux(Pahs, xT)

L = hcat(Matrix(1.0I,2*n,2*n), zeros(2*n,2*n))
m, d = size(L)
Σ = Matrix(Σdiagel*I,m,m)
=#
