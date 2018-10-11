# Library

## Important concepts

```@docs
ContinuousTimeProcess{T}
SamplePath{T}
Bridge.GSamplePath
valtype
Bridge.outertype
```

## Ordinary differential equations and quadrature

```@docs
Bridge.ODESolver
solve!
Bridge.solvebackward!
Bridge.R3
Bridge.BS3
LeftRule
Bridge.fundamental_matrix
```


## Brownian motion

```@autodocs
Modules = [Bridge]
Pages = ["/wiener.jl"]
```

## Stochastic differential equations

```@docs
Bridge.a
sample
sample!
quvar
bracket
ito
girsanov
lp
llikelihood
solve
EulerMaruyama
Euler
StochasticRungeKutta
StochasticHeun
Bridge.NoDrift
```

## In place solvers
```@docs
Bridge.R3!
Bridge.σ!
Bridge.b!
```

## Levy processes
```@docs
GammaProcess
GammaBridge
Bridge.ExpCounting
Bridge.CompoundPoisson
Bridge.nu
Bridge.uniform_thinning!
```

## Poisson processes
```@docs
ThinningAlg
InhomogPoisson
```

## Bessel processes
```@docs
Bridge.Bessel{N}
Bridge.Bessel3Bridge
Bridge.BesselProp
```

## Miscellaneous

```@docs
Bridge.endpoint!
Bridge.inner
Bridge.cumsum0
Bridge.mat
Bridge.outer
CSpline
Bridge.integrate
Bridge.logpdfnormal
Bridge.runmean
Bridge.PSD
Bridge.Gaussian
Bridge.refine
Bridge.quaternion
Bridge._viridis
Bridge.supnorm
Bridge.posterior
```

## Online statistics

Online updating of the tuple `state = (m, m2, n)` where

`m` - `mean(x[1:n])`

`m2` - sum of squares of differences from the current mean, ``\textstyle\sum_{i=1}^n (x_i - \bar x_n)^2``

`n` - number of iterations

```@docs
mcstart
mcnext
mcband
mcbandmean
Bridge.mcstats
Bridge.mcmarginalstats
Bridge.OnlineStat
```

## Linear Processes

```@docs
LinPro
Bridge.Ptilde
Bridge.LinearNoiseAppr
Bridge.LinearAppr
Bridge.LinProBridge
```


## Bridges

```@docs
GuidedProp
Bridge.GuidedBridge
Bridge.PartialBridge
Bridge.PartialBridgeνH
BridgeProp
Bridge.Mdb
Bridge.Vs
Bridge.gpV!
Bridge.r
Bridge.gpHinv!
Bridge.gpupdate
```

## Unsorted

```@docs
LocalGammaProcess
Bridge.compensator0
Bridge.compensator
Bridge.θ
Bridge.soft
Bridge.tofs
Bridge.dotVs
Bridge.SDESolver
Bridge.Increments
Bridge.sizedtype
Bridge.piecewise
Bridge.aeuler
Bridge.MeanCov
Bridge.upsample
Bridge.viridis
Bridge.rescale
```
