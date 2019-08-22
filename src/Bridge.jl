module Bridge
export ContinuousTimeProcess, SamplePath, VSamplePath, GSamplePath
export stack

export LinPro, Wiener, WienerBridge, CSpline
export sample, sample!, .., quvar, ito, bracket, lp, llikelihood, transitionprob, girsanov


export BridgeProp, DHBridgeProp, FilterProp, PBridgeProp, GuidedProp, GuidedBridge, innovations, innovations!, lptilde
export ubridge!, ubridge

export ullikelihood, ullikelihoodtrapez, uinnovations!, ubridge

# LinearProcess
export LinearProcess

# Levy
export LevyProcess, GammaProcess, GammaBridge, VarianceGammaProcess, LocalGammaProcess

# Poisson
export ThinningAlg, InhomogPoisson

# mclog
export mcstart, mcnext, mcbandmean, mcband

# euler
export SDESolver, Euler, BridgePre, EulerMaruyama, EulerMaruyama!, StochasticHeun,
    StochasticRungeKutta, bridge!, bridge, solve, solve!

# ode
export solve, solve!, solvebackward!, ODESolver, BS3

# guip
export LeftRule

using Random
using LinearAlgebra
using Statistics

include("chol.jl")

using StaticArrays
using Distributions
using Colors
using Trajectories
include("unroll1.jl")


import Base: rand
import Random: rand!

import Distributions: sample, sample!

function Btilde
end
function βtilde
end
function mcsvd3
end
function visualize_uncertainty
end
function B!
end
function a!
end

"""
    b!(t, y, tmp1, P)

Compute drift ``b`` in `y` (without factor ``Δt``, modifying `tmp1`.
"""
function b!
end

"""
    σ!(t, y, Δw, tmp2, P)

Compute stochastic increment at `y`, ``σ Δw``, modifying `tmp2`.
"""
function σ!
end

P(Po) = Po.Target
Pt(Po) = Po.Pt

target(Po) = Po.Target
auxiliary(Po) = Po.Pt


include("expint.jl")
#include("setcol.jl")

include("fsa.jl")
include("gaussian.jl")

include("types.jl")
include("sizedtype.jl")
include("misc.jl")

#_b((i,s)::IndexedTime, x, P) = b(s, x, P)
#_b!((i,s)::IndexedTime, x, tmp, P) = b!(s, x, tmp, P)
_b((i,s), x, P) = b(s, x, P)
_b!((i,s), x, tmp, P) = b!(s, x, tmp, P)

#btilde((i,s)::IndexedTime, x, P) = btilde(s, x, P)
#a((i,s)::IndexedTime, x, P) = a(s, x, P)
r((i,s)::IndexedTime, x, P) = r(s, x, P)
H((i,s)::IndexedTime, x, P) = H(s, x, P)

include("cspline.jl")
include("wiener.jl")
include("ellipse.jl")
include("ode.jl")
include("gode.jl")
include("lyap.jl")

include("ode!.jl")
include("diffusion.jl")
include("linpro.jl")
include("guip.jl")
include("guip!.jl")

include("partialbridge.jl")
include("partialbridgenuH.jl")
include("partialbridgen!.jl")

include("euler.jl")
include("sde.jl")

include("sde!.jl")
include("levy.jl")
include("poisson.jl")

include("timechange.jl")
include("uncertainty.jl")
include("mclog.jl")

include("bessel.jl")

include("Models.jl")

include("recipes.jl")

include("deprecated.jl")

end
