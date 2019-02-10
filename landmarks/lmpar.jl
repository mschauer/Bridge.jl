# reminder, to type H*, do H\^+
#outdir="output/landmarks/"
cd("/Users/Frank/.julia/dev/Bridge/landmarks")

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
model = models[2]
TEST = false#true

const d = 2

n = 40 # nr of landmarks

θ = 0#-π/20# π/6 0#π/5  # angle to rotate endpoint

σobs = 10^(-2)   # noise on observations
println(model)

T = 0.5#1.0#0.5
t = 0.0:0.01:T  # time grid

#Random.seed!(5)


include("state.jl")
include("msmodel.jl")
include("ahsmodel.jl")
include("bothmodels.jl")
include("patches.jl")
include("lmguiding.jl")

### Specify landmarks models

a = 1.5 ; λ = 0.0; #= not the lambda of noise fields  =# γ = 5.0

Pms = MarslandShardlow(a, γ, λ, n)

if true
    db = 3.0 # domainbound
    # specify locations noisefields
    r1 = (-db:0.5:db) #+ 0.1rand(5)
    r2 = (-db:0.5:db) #+ 0.1rand(5)
    nfloc = Point.(collect(product(r1, r2)))[:]
    # scaling parameter of noisefields
    nfscales = [0.3Point(1.0, 1.0) for x in nfloc]
    nfstd = 2.0 # tau , variance of noisefields
else
    db = 2.0 # domainbound
    # specify locations noisefields
    r1 = (-db:1.0:db) #+ 0.1rand(5)
    r2 = (-db:1.0:db) #+ 0.1rand(5)
    nfloc = Point.(collect(product(r1, r2)))[:]
    # scaling parameter of noisefields
    nfscales = [5Point(1.0, 1.0) for x in nfloc]
    nfstd = 1.0 # tau , variance of noisefields
end

nfs = [Noisefield(δ, λ, nfstd) for (δ, λ) in zip(nfloc, nfscales)]

Pahs = Landmarks(a, λ, n, nfs)
###

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

# specify initial landmarks configuration
q0 = [Point(1.5cos(t), sin(t)) for t in (0:(2pi/n):2pi)[1:n]]  #q0 = circshift(q0, (1,))
p_ = Point(0.1, 0.1)
p0 = [5p_ for i in 1:n]  #
#p0 = [randn(Point) for i in 1:n]
x0 = State(q0, p0)


#Random.seed!(1234)

X = SamplePath(t, [copy(x0) for s in t])
if model == :ms
    @time solve!(EulerMaruyama!(), X, x0, W, P)
else
    @time solve!(EulerMaruyama!(), X, x0, W, P)
    # still needs to be implemented
    #@time solve!(StratonovichHeun!(), X, x0, W, P)
end
extractcomp(v,i) = map(x->x[i], v)

# observe positions without noise
v0 = q(X.yy[1])
rot =  SMatrix{2,2}(cos(θ), sin(θ), -sin(θ), cos(θ))
vT = [rot * X.yy[end].q[i] for i in 1:P.n ]

out = [Any[X.tt[i], [X.yy[i][CartesianIndex(c, k)][l] for l in 1:d, c in 1:2]..., "point$k"] for k in 1:n, i in eachindex(X.tt) ][:]
df = DataFrame(time=extractcomp(out,1),pos1=extractcomp(out,2),pos2=extractcomp(out,3),mom1=extractcomp(out,4),mom2=extractcomp(out,5),pointID=extractcomp(out,6))
#Any["X.tt[$i]", ["X.yy[$i][CartesianIndex($c, $k)][$l]" for l in 1:d, k in 1:n, c in 1:2]...]

if model == :ms
    titel = "Marsland-Shardlow model, "
else
    titel = "Arnaudon-Holm-Sommer model, "
end
titel = titel * string(n)*" landmarks"

dfT = DataFrame(pos1=extractcomp(vT,1), pos2=extractcomp(vT,2))
df0= DataFrame(pos1=extractcomp(v0,1), pos2=extractcomp(v0,2))

@rput titel
@rput df
@rput dfT
@rput df0
R"""
library(tidyverse)
#df %>% dplyr::select(time,pos1,pos2,pointID) %>%
     ggplot() +
      geom_path(data=df,aes(x=pos1,pos2,group=pointID,colour=time)) +
      geom_point(data=dfT, mapping=aes(x=pos1,y=pos2),colour='orange',size=0.7) +
      geom_point(data=df0, mapping=aes(x=pos1,y=pos2),colour='black',size=0.7)+
      theme_minimal() + xlab('horizontal position') +
      ylab('vertical position') + ggtitle(titel)
"""

R"""
df %>% dplyr::select(time,mom1,mom2,pointID) %>%
    filter(pointID %in% c("point1","point2","point3","point40","point41","point42")) %>%
     ggplot(aes(x=mom1,mom2,group=pointID,colour=time)) + geom_path() +
     facet_wrap(~pointID) + theme_minimal()
"""

tc(t,T) = t.*(2-t/T)
tt_ =  t#tc(t,T)# 0:dtimp:(T)
####################
# solve backward recursion on [0,T]
#L = deepmat( [(i==j)*one(Unc) for i in 1:2:2n, j in 1:2n])
#Σ = Diagonal(σobs^2*ones(n*d))
L = [(i==j)*one(Unc) for i in 1:2:2n, j in 1:2n]
Σ = [(i==j)*σobs^2*one(Unc) for i in 1:n, j in 1:n]
Pmsaux = MarslandShardlowAux(Pms, State(vT, zero(vT)))
cheat = false
if cheat
    Pahsaux = LandmarksAux(Pahs, X.yy[end])
else
#    Pahsaux = LandmarksAux(Pahs, State(vT, zero(vT)))
#    Pahsaux = LandmarksAux(Pahs, State(vT, [Point(0.001,0.001) for i in 1:Pahs.n]))
    Pahsaux = LandmarksAux(Pahs, State(vT, rand(Point,Pahs.n)))
end

if model == :ms
    Paux = Pmsaux
else
    Paux = Pahsaux
end

# initialise Lt and M⁺t
Lt =  [copy(L) for s in tt_]
Mt⁺ = [copy(Σ) for s in tt_]
@time (Lend, Mend⁺) =  guidingbackwards!(Lm(), t, (Lt, Mt⁺), Paux, (L, Σ))

# issymmetric(deepmat(Bridge.a(0,Paux)))
# isposdef(deepmat(Bridge.a(0,Paux)))
# map(x->minimum(eigen(deepmat(x)).values),Mt⁺)
#Mt = map(X -> deepmat2unc(inv(deepmat(X))),Mt⁺)
Mt = map(X -> InverseCholesky(lchol(X)),Mt⁺)


Q = GuidedProposal!(P, Paux, tt_, Lt, Mt, vT)

#xinit = State(v0, zero(v0))
#xinit = State(v0, rand(Point,P.n))
xinit = x0
winit = zeros(StateW, dwiener)
XX = SamplePath(tt_, [copy(xinit) for s in tt_])
WW = SamplePath(tt_, [copy(winit) for s in tt_])
sample!(WW, Wiener{Vector{StateW}}())

Bridge.solve!(EulerMaruyama!(), XX, xinit, WW, Q)

#### plotting
outg = [Any[XX.tt[i], [XX.yy[i][CartesianIndex(c, k)][l] for l in 1:d, c in 1:2]..., "point$k"] for k in 1:n, i in eachindex(XX.tt) ][:]
dfg = DataFrame(time=extractcomp(outg,1),pos1=extractcomp(outg,2),pos2=extractcomp(outg,3),mom1=extractcomp(outg,4),mom2=extractcomp(outg,5),pointID=extractcomp(outg,6))
#Any["X.tt[$i]", ["X.yy[$i][CartesianIndex($c, $k)][$l]" for l in 1:d, k in 1:n, c in 1:2]...]

dfT = DataFrame(pos1=extractcomp(vT,1), pos2=extractcomp(vT,2))
df0= DataFrame(pos1=extractcomp(v0,1), pos2=extractcomp(v0,2))


if model == :ms
    titel = "Marsland-Shardlow model, "
else
    titel = "Arnaudon-Holm-Sommer model, "
end
titel = titel * string(n)*" landmarks"

@rput titel
@rput dfg
@rput dfT
@rput df0
@rput T
R"""
library(tidyverse)
T_ <- T-0.001
#T_ <- T
#T_ = 0.05
dfsub <- df %>% dplyr::select(time,pos1,pos2,pointID) %>% filter(time<T_)
dfsubg <- dfg %>% dplyr::select(time,pos1,pos2,pointID) %>% filter(time<T_)

sub <- rbind(dfsub,dfsubg)
sub$fg <- rep(c("forward","guided"),each=nrow(dfsub))

g <- ggplot() +
      geom_path(data=sub, mapping=aes(x=pos1,pos2,group=pointID,colour=time)) +
      geom_point(data=dfT, mapping=aes(x=pos1,y=pos2),colour='orange',size=0.7) +
      geom_point(data=df0, mapping=aes(x=pos1,y=pos2),colour='black',size=0.7)+
      facet_wrap(~fg) +
      theme_minimal() + xlab('horizontal position') +
      ylab('vertical position') + ggtitle(titel)
show(g)
"""

if false
R"""
dfg %>% dplyr::select(time,mom1,mom2,pointID) %>%
    filter(pointID %in% c("point1","point2","point3","point40","point41","point42")) %>%
     ggplot(aes(x=mom1,mom2,group=pointID,colour=time)) + geom_path() +
     facet_wrap(~pointID) + theme_minimal()
"""



    R"""
    library(tidyverse)
    dfsub <- df %>% dplyr::select(time,pos1,pos2,pointID) %>% filter(time<0.99)
    dfsubg <- dfg %>% dplyr::select(time,pos1,pos2,pointID) %>% filter(time<0.99)

    sub <- rbind(dfsub,dfsubg)
    sub$fg <- rep(c("forward","guided"),each=nrow(dfsub))

    ggplot() +
          geom_path(data=sub, mapping=aes(x=pos1,pos2,group=pointID,colour=fg)) +
          geom_point(data=dfT, mapping=aes(x=pos1,y=pos2),colour='orange',size=0.7) +
          geom_point(data=df0, mapping=aes(x=pos1,y=pos2),colour='black',size=0.7)+
          #facet_wrap(~fg) +
          theme_minimal() + xlab('horizontal position') +
          ylab('vertical position') + ggtitle(titel)
    """
end

if model==:ms
    @time llikelihood(LeftRule(), XX, Q; skip = 0)  # won't work for AHS because matrix multilication for Htilde is not defined yet
end
