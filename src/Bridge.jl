module Bridge
export ContinuousTimeProcess, SamplePath, Wiener, WienerBridge, sample, sample!, .., euler,euler!, eulerb,eulerb!, quvar, ito, bracket, lp, llikelihood,transitionprob, girsanov, LevyProcess, GammaProcess, GammaBridge, VarianceGammaProcess
export BridgeProp, DHBridgeProp, FilterProp, PBridgeProp, innovations, innovations!, lptilde


using Distributions
import Distributions: sample, sample!
using FixedSizeArrays

include("diagonal.jl")
include("misc.jl")

include("types.jl")

include("wiener.jl")
include("ellipse.jl")
include("euler.jl")
include("diffusion.jl")
include("guip.jl")
include("levy.jl")

export FixedDiagonal

end
