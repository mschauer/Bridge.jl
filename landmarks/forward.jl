# reminder, to type H*, do H\^+
#cd("/Users/Frank/.julia/dev/Bridge")
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

discrmethods = [:ralston, :lowrank, :psd]
discrmethod = discrmethods[2]

obsschemes =[:full, :partial]
obsscheme = obsschemes[2]

const d = 2

n = 40 # nr of landmarks
ldim = 2n*d   # dimension of low-rank approximation to H\^+

cheat =  false#true#false # if cheat is true, then we initialise at x0 (true value) and
# construct the guiding term based on xT (true value)

θ = 6π/40# π/6 0#π/5  # angle to rotate endpoint

ϵ = 0.001   # parameter for initialising Hend⁺
obsnoise = 0.001   # noise on observations



println(model)
println(discrmethod)
println(obsscheme)

T = 0.5#1.0#0.5
t = 0.0:0.005:T  # time grid

#Random.seed!(5)


include("state.jl")
include("msmodel.jl")
include("ahsmodel.jl")
include("patches.jl")
include("guiding.jl")
#include("lyap.jl")
include("LowrankRiccati.jl")
using .LowrankRiccati


### Specify landmarks models
a = 1.0 ; λ = 0.5; #= not the lambda of noise fields  =# γ = 5.0

Pms = MarslandShardlow(a, γ, λ, n)

if true
    # specify locations noisefields
    r1 = (-2.0:0.5:2.0) #+ 0.1rand(5)
    r2 = (-2.0:0.5:2.0) #+ 0.1rand(5)
    nfloc = Point.(collect(product(r1, r2)))[:]
    # scaling parameter of noisefields
    nfscales = [0.1Point(1.0, 1.0) for x in nfloc]
    nfstd = 2.05 # tau , variance of noisefields
else
    # specify locations noisefields
    r1 = (-2.0:0.5:2.0) #+ 0.1rand(5)
    r2 = (-2.0:0.5:2.0) #+ 0.1rand(5)
    nfloc = Point.(collect(product(r1, r2)))[:]
    # scaling parameter of noisefields
    nfscales = [2.5Point(1.0, 1.0) for x in nfloc]
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
q0 = [Point(cos(t), sin(t)) for t in (0:(2pi/n):2pi)[1:n]]  #q0 = circshift(q0, (1,))
p_ = 5*Point(0.1, 0.1)
p0 = [p_ for i in 1:n]  #
p0 = [randn(Point) for i in 1:n]
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

if TEST
    ####### extracting components
    X.tt   # gives all times
    X.yy   # gives an array of States (the state at each time)
    X.yy[1] # gives the state at first time

    q(X.yy[1]) # gives the array of positions of all points
    X.yy[1].q # equivalent

    p(X.yy[1]) # gives the array of momenta of all points
    X.yy[1].p # equivalent

    p(X.yy[1],10)   # at first time, extract momentum vector of landmark nr 10
    q(X.yy[23],10)  # extract position at time 23, for landmark nr 10

    map(x->norm(x), X.yy[1].q) # verifies that all points are on the unit circle
    ham = [hamiltonian(X.yy[i],Pms) for i in 1:length(t)]
    print(ham)
    #############
end


out = [Any[X.tt[i], [X.yy[i][CartesianIndex(c, k)][l] for l in 1:d, c in 1:2]..., "point$k"] for k in 1:n, i in eachindex(X.tt) ][:]
df = DataFrame(time=extractcomp(out,1),pos1=extractcomp(out,2),pos2=extractcomp(out,3),mom1=extractcomp(out,4),mom2=extractcomp(out,5),pointID=extractcomp(out,6))
#Any["X.tt[$i]", ["X.yy[$i][CartesianIndex($c, $k)][$l]" for l in 1:d, k in 1:n, c in 1:2]...]

if model == :ms
    titel = "Marsland-Shardlow model, "
else
    titel = "Arnaudon-Holm-Sommer model, "
end
titel = titel * string(n)*" landmarks"

@rput titel
@rput df
R"""
library(tidyverse)
df %>% dplyr::select(time,pos1,pos2,pointID) %>%
     ggplot(aes(x=pos1,pos2,group=pointID,colour=time)) +
      geom_path() +
      theme_minimal() + xlab('horizontal position') +
      ylab('vertical position') + ggtitle(titel)
"""

R"""
df %>% dplyr::select(time,mom1,mom2,pointID) %>%
    filter(pointID %in% c("point1","point2","point3","point40","point41","point42")) %>%
     ggplot(aes(x=mom1,mom2,group=pointID,colour=time)) + geom_path() +
     facet_wrap(~pointID) + theme_minimal()
"""

# observe positions without noise
v0 = q(X.yy[1])
rot =  SMatrix{2,2}(cos(θ), sin(θ), -sin(θ), cos(θ))
vT = [rot * q(X.yy[end])[i] for i in 1:P.n ]
#vT = q(X.yy[end])



#dtimp = 0.01 # mesh width for imputation
tc(t,T) = t.*(2-t/T)
tt_ =  t#tc(t,T)# 0:dtimp:(T)

####################
# solve backward recursion on [0,T]




if obsscheme==:partial
  L = deepmat( [(i==j)*one(Unc) for i in 1:2:2n, j in 1:2n])
  Σ = Diagonal(obsnoise*ones(n*d))
  Pmsaux = MarslandShardlowAux(Pms, State(vT, zero(vT)))
  if cheat
      Pahsaux = LandmarksAux(Pahs, X.yy[end])
  else
      Pahsaux = LandmarksAux(Pahs, State(vT, zero(vT)))
      #Pahsaux = LandmarksAux(Pahs, State(vT, rand(Point,Pahs.n)))
      #Pahsaux = LandmarksAux(Pahs, State(vT,  [Point(1,1) for i in 1:Pahs.n]))
      #Pahsaux = LandmarksAux(Pahs, X.yy[end])
      #Pahsaux = LandmarksAux(Pahs, State(vT, 0.0*X.yy[end].p))
  end
end
if obsscheme==:full
  L = deepmat( [(i==j)*one(Unc) for i in 1:2n, j in 1:2n])
  Σ = Diagonal(obsnoise*ones(2n*d))
  Pmsaux = MarslandShardlowAux(Pms,X.yy[end])
  Pahsaux = LandmarksAux(Pahs, X.yy[end])
  v0 = vec(X.yy[1])
  vT = vec(X.yy[end])
end

if model == :ms
    Paux = Pmsaux
else
    Paux = Pahsaux
end

Hend⁺ = [(i==j)*one(Unc)/ϵ for i in 1:2n, j in 1:2n]
#### perform gpupdate step
νend , Hend⁺ = gpupdate(Paux.xT,Hend⁺, Σ, L, vT)
# L, and Σ are ordinary matrices, vT an array of Points
# νend is a state , Hend⁺ a  UncMat

# initialise νt
νt =  [copy(νend) for s in tt_]


if discrmethod==:lowrank
    M0 = eigen(deepmat(Hend⁺))
    largest = sortperm(M0.values)[end-ldim+1:end]
    Send = Matrix(Diagonal(M0.values[largest]))
    Uend = M0.vectors[:,largest]
#    println(deepmat(Hend⁺)-Uend*Send*Uend')
    St = [copy(Send) for s in tt_]
    Ut = [copy(Uend) for s in tt_]
    @time νend, (Send, Uend) = bucybackwards!(LRR(), t, νt, (St, Ut), Paux, νend, (Send, Uend))
    Hend⁺ = deepmat2unc(Uend * Send * Uend')
    #map(x->isposdef(x),St)
#    Ht = map((S,U) -> LowRank(cholesky(Hermitian(S)),U),St,Ut)
     Ht = map((S,U) -> LowRank(S,U), St,Ut)
end
if discrmethod==:ralston
    H⁺t = [copy(Hend⁺) for s in tt_]
    @time νend , Hend⁺ = bucybackwards!(Bridge.R3!(), tt_, νt, H⁺t, Paux, νend, Hend⁺)
    println(map(x->isposdef(deepmat(x)),H⁺t))
    Ht = map(H⁺ -> InverseCholesky(lchol(H⁺)),H⁺t)
end
if discrmethod==:psd
    H⁺t = [copy(Hend⁺) for s in tt_]
    @time νend , Hend⁺ = bucybackwards!(Lyap(), tt_, νt, H⁺t, Paux, νend, Hend⁺)
    println(map(x->isposdef(deepmat(x)),H⁺t))
    #Ht = map(H⁺ -> InverseCholesky(lchol(Hermitian(H⁺))),H⁺t)
    Ht = map(H⁺ -> InverseCholesky(lchol(H⁺)),H⁺t)
    #map(H⁺ -> lchol(Hermitian(H⁺)),H⁺t)
end




Q = GuidedProposal!(P, Paux, tt_, νt, Ht)

νstart , Hstart⁺ = gpupdate(νend , Hend⁺, Σ, L, v0)
#νstart , Hstart⁺ = gpupdate(νend , Hend⁺, Σ, L, q(XX.yy[1]))

xinit = cheat ? x0 : νstart  # or xinit ~ N(νstart, Hstart⁺)
winit = zeros(StateW, dwiener)
XX = SamplePath(tt_, [copy(xinit) for s in tt_])
WW = SamplePath(tt_, [copy(winit) for s in tt_])
sample!(WW, Wiener{Vector{StateW}}())

Bridge.solve!(EulerMaruyama!(), XX, xinit, WW, Q)

hcat(mean(X.yy[end].p), mean(XX.yy[end].p))
# redo with Paux.xT = XX.yy[end]
 hcat(X.yy[end].p, XX.yy[end].p, Paux.xT.p)

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
T_ <- T-0.0001
#T_ <- 0.8
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
@time llikelihood(LeftRule(), XX, Q; skip = 0)  # won't work for AHS because matrix multilication for Htilde is not defined yet
