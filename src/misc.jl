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

"""
    inner(x[, y])
Short-hand for quadratic form x'x (or x'y).
"""
inner(x) = dot(x,x)
inner(x,y) = dot(x,y)


"""
    mat(X::SamplePath{SVector}) 
    mat(yy::Vector{SVector})

Reinterpret `X` or `yy` to an array without change in memory.
"""
mat(X::SamplePath{SVector{d,T}}) where {d,T} = reshape(reinterpret(T, X.yy), d, length(X.yy))
mat(yy::Vector{SVector{d,T}}) where {d,T} = reshape(reinterpret(T, yy), d, length(yy))

unmat(A::Matrix{T}) where {T} = reinterpret(SVector{size(A, 1),T}, A[:])
unmat(::Type{SVector{d,T}}, A::Matrix{T}) where {d,T} = reinterpret(SVector{d,T}, A[:])
