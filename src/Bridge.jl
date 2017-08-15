__precompile__()

module Bridge
export ContinuousTimeProcess, SamplePath, VSamplePath
export LinPro, Wiener, WienerBridge, CSpline
export sample, sample!, .., euler, euler!, quvar, ito, bracket, lp, llikelihood, transitionprob, girsanov


export BridgeProp, DHBridgeProp, FilterProp, PBridgeProp, GuidedProp, innovations, innovations!, lptilde
export ubridge!, ubridge
export thetamethod, thetamethod!, thetainnovations!, thetainnovations, heun, heun!

export ullikelihood, ullikelihoodtrapez, uinnovations!, ubridge

# Levy
export LevyProcess, GammaProcess, GammaBridge, VarianceGammaProcess, LocalGammaProcess

# mclog
export mcstart, mcnext, mcbandmean, mcband

# euler
export SDESolver, Euler, bridge!, bridge

# ode
export solve!, solvebackward!, ODESolver, RS3

# guip
export LeftRule

using Distributions
using StaticArrays
using Compat

import Base.rand
import Distributions: sample, sample!

include("expint.jl")

include("fsa.jl")
include("gaussian.jl")

include("types.jl")
include("misc.jl")

include("cspline.jl")
include("wiener.jl")
include("ellipse.jl")
include("ode.jl")
include("euler.jl")
include("diffusion.jl")
include("guip.jl")
include("levy.jl")
include("linpro.jl")
include("timechange.jl")
include("mclog.jl")
include("deprecated.jl")

end
