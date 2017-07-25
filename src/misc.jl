export supnorm, @_isdefined
"""
    cumsum0

Cumulative sum starting at 0, 
"""
function cumsum0(dx::Vector)
        n = length(dx) + 1
        x = similar(dx, n)
        x[1] = 0.0      
        for i in 2:n
                x[i] = x[i-1] + dx[i-1] 
        end
        x
end

supnorm(x) = sum(abs.(x))

macro _isdefined(var)
    quote
        try local _ = $(esc(var))
            true
        catch err
            isa(err, UndefVarError) ? false : rethrow(err)
        end
    end
end

if isempty(methods(chol, (UniformScaling,)))
    include("chol.jl")
end

"""
    outer(x[, y])
Short-hand for quadratic form xx' (or xy').
"""
outer(x) = x*x'
outer(x,y) = x*y'
