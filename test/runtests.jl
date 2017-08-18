include(joinpath("..", "docs", "make.jl"))

srand(joinpath(@__DIR__,"SEED"),1)

include("wiener.jl")
include("diffusion.jl")
include("euler.jl")
include("misc.jl")
include("VHK.jl")
include("guip.jl")
include("linpro.jl")
include("timechange.jl")
include("uniformscaling.jl")
include("gamma.jl")
include("with_srand.jl") #run last