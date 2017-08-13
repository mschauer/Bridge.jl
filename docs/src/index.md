# Bridge.jl

Documentation for Bridge.jl

## Important concepts

```@docs
ContinuousTimeProcess{T}
SamplePath{T}
valtype
```

## Ordinary differential equations

```@docs
solve!
Bridge.R3
Bridge.BS3
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
```

## Miscallenious
```@docs
Bridge.endpoint!
```

## Unsorted
```@docs
Bridge.ODESolver
mcbandmean 
Bridge.cumsum0 
LocalGammaProcess
Bridge.Ptilde
GuidedProp
Bridge.compensator0 
mcstart 
CSpline 
Bridge.tofs
Bridge.r
Bridge.θ 
Bridge.gpK! 
mcnext 
Bridge.compensator 
Bridge.outer 
Bridge.integrate 
Bridge.soft
Bridge.nu 
bridge
Bridge.Vs
Bridge.logpdfnormal
Bridge.mdb!
euler! 
GammaProcess
Bridge.dotVs
Bridge.mdb 
euler 
Bridge.mat 
thetamethod 
mcband 
LinPro
```