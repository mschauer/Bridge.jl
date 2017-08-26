# Library

## Important concepts

```@docs
ContinuousTimeProcess{T}
SamplePath{T}
valtype
```

## Ordinary differential equations and quadrature

```@docs
Bridge.ODESolver
solve!
Bridge.R3
Bridge.BS3
LeftRule
```


## Brownian motion

```@autodocs
Modules = [Bridge]
Pages = ["/wiener.jl"]
```

## Stochastic differential equations

```@docs
sample
sample!
quvar
bracket
ito
girsanov
lp
llikelihood
solve
euler
euler!
EulerMaruyama
Euler
StochasticRungeKutta
StochasticHeun
```

## Levy processes
```@docs
GammaProcess
GammaBridge
Bridge.ExpCounting
Bridge.CompoundPoisson
Bridge.nu 
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
```

## Linear Processes

```@docs
LinPro
Bridge.Ptilde
```


## Bridges

```@docs
GuidedProp
Bridge.GuidedBridge
BridgePre
Bridge.Mdb
bridge
bridge!
Bridge.Vs
Bridge.r
Bridge.gpHinv!
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
```