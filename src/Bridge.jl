module Bridge
export ContinuousTimeProcess, SamplePath
export LinPro, PTilde, Wiener, WienerBridge, CSpline
export sample, sample!, .., euler, euler!, rungekuttab, rungekuttab!, quvar, ito, bracket, lp, llikelihood, transitionprob, girsanov
export bridge!, bridge
export LevyProcess, GammaProcess, GammaBridge, VarianceGammaProcess
export BridgeProp, DHBridgeProp, FilterProp, PBridgeProp, GuidedProp, UProp, innovations, innovations!, lptilde
export ubridge!, ubridge
export thetamethod, thetamethod!, thetainnovations!, thetainnovations, heun, heun!

export ullikelihood, ullikelihoodtrapez, uinnovations!, ubridge

using Distributions
import Base.rand
import Distributions: sample, sample!
using StaticArrays

include("diagonal.jl")
include("misc.jl")
include("fsa.jl")
include("gaussian.jl")

include("types.jl")
include("cspline.jl")
include("wiener.jl")
include("ellipse.jl")
include("euler.jl")
include("diffusion.jl")
include("guip.jl")
include("levy.jl")
include("linpro.jl")
include("timechange.jl")

export SDiagonal

end
