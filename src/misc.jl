export supnorm, @_isdefined


"""
    refine(tt, n) 

Refine range by decreasing stepsize by a factor `n`.
"""
refine(tt, n) =  first(tt):(Base.step(tt)/n):last(tt)


"""
    runmean(x)

Running mean of the vector `x`.
"""
runmean(x, cx = cumsum(x)) = [cx[n]/n for n in 1:length(x)]

function runmean(xx::Matrix) 
    yy = copy(xx) / 1
    m = 0 * (copy(xx[1,:])/1)
    for i in 1:size(yy, 1)
        m[:] = m + (xx[i, :] - m)/i
        yy[i, :] = m
    end
    yy
end

"""
    cumsum0(x)

Cumulative sum starting at 0 such that `cumsum0(diff(x)) ≈ x`.
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
    mat(yy::Vector{SVector})

Reinterpret `X` or `yy` to an array without change in memory.
"""
mat(yy::Vector{SVector{d,T}}) where {d,T} = reshape(reinterpret(T, yy), d, length(yy))

unmat(A::Matrix{T}) where {T} = reinterpret(SVector{size(A, 1),T}, A[:])
unmat(::Type{SVector{d,T}}, A::Matrix{T}) where {d,T} = reinterpret(SVector{d,T}, A[:])


"""
    quaternion(m::SMatrix{3,3})

Compute the (rotation-) quarternion of a 3x3 rotation matrix. Useful to create
isodensity ellipses from spheres in GL visualizations.
"""
function quaternion(m)
    qw = √(1 + m[1,1] + m[2,2] + m[3,3])/2
    qx = (m[3,2] - m[2,3])/(4qw)
    qy = (m[1,3] - m[3,1])/(4qw)
    qz = (m[2,1] - m[1,2])/(4qw)
    SVector{4}(qx,qy,qz,qw)
end