using RecipesBase, Distributions

@recipe function plot(X::SamplePath)
    X.tt, X.yy
end

@recipe function plot(X::VSamplePath)
    X.tt, X.yy'
end

@recipe function plot(X::Bridge.EstSamplePath{Float64}; quantile = 0.95)
    q = Distributions.quantile(Normal(), 1 - (1 - quantile)/2)
    ribbon --> q*sqrt.(X.vv)
    X.tt, X.yy
end

#@recipe function plot(X::Bridge.SamplePathBand)
#    ribbon --> (X.uu .- X.ll)
#    X.tt, (X.uu .+ X.ll)./2
#end