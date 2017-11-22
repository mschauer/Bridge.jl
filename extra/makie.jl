using Makie
import Makie.to_positions

to_positions(S::Makie.Scene, X::Bridge.SamplePath) = to_positions(S, X.yy)
