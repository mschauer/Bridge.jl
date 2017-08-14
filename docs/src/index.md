# Bridge.jl

Documentation for Bridge.jl

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
euler
euler!
thetamethod
```

## Levy processes
```@docs
GammaProcess
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
bridge
Bridge.Vs
Bridge.mdb 
Bridge.mdb!
Bridge.r
Bridge.gpK! 
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
```