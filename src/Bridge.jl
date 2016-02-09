module Bridge
export ContinuousTimeProcess, SamplePath
export LinPro, PTilde, Wiener, WienerBridge, CSpline
export sample, sample!, .., euler,euler!, eulerb,eulerb!, quvar, ito, bracket, lp, llikelihood, transitionprob, girsanov
export LevyProcess, GammaProcess, GammaBridge, VarianceGammaProcess
export BridgeProp, DHBridgeProp, FilterProp, PBridgeProp, GuidedProp, innovations, innovations!, lptilde



using Distributions
import Base.rand
import Distributions: sample, sample!
using FixedSizeArrays

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

export FixedDiagonal

end
