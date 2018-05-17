__precompile__()

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

using Distributions
using StaticArrays
using Compat

import Base.rand
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
function bi!
end
function ri!
end
"""
    σ!(t, y, Δw, tmp2, P)

Compute stochastic increment at `y`, ``σ Δw``, modifying `tmp2`.
"""
function σ!
end

function P
end
function Pt
end


hasbi(::Any) = false
hasai(::Any) = false

include("expint.jl")

include("fsa.jl")
include("gaussian.jl")

include("types.jl")
include("sizedtype.jl")
include("misc.jl")

include("cspline.jl")
include("wiener.jl")
include("ellipse.jl")
include("ode.jl")
include("diffusion.jl")
include("linpro.jl")
include("guip.jl")
include("euler.jl")
include("levy.jl")
include("poisson.jl")

include("timechange.jl")
include("uncertainty.jl")
include("mclog.jl")

include("Models.jl")

include("recipes.jl")

include("deprecated.jl")

end
