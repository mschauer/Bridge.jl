using RecipesBase, Distributions

@recipe function plot(X::SamplePath)
    X.tt, X.yy
end

@recipe function plot(X::VSamplePath)
    X.tt, X.yy'
end

@recipe function plot(X::EstSamplePath{Float64}; quantile = 0.95)
    q = quantile(Normal(), 1 - (1 - quantile)/2)
    u = X.yy + q*sqrt.(X.vv)
    l = X.yy - q*sqrt.(X.vv)
    fillrange --> l
    fillalpha --> 0.5 
    X.tt, u
end