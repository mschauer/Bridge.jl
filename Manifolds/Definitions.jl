using ForwardDiff
using LinearAlgebra

abstract type Manifold end
abstract type SDEForm end

struct Ito <: SDEForm end
struct Stratonovich <: SDEForm end

const ℝ{N} = SVector{N, Float64}
const IndexedTime = Tuple{Int64,Float64}
outer(x) = x*x'
outer(x,y) = x*y'

"""
    EmbeddedManifold creates a manifold ``M = f^{-1}({0})`` where
    ``f`` should be a smooth function ``ℝ^N → ℝ``
"""

abstract type EmbeddedManifold <: Manifold end

"""
    Settings for the sphere 𝕊²
"""

struct Sphere{T<:Real} <: EmbeddedManifold
    R::T

    function Sphere(R::T) where {T<:Real}
        if R<=0
            error("R must be positive")
        end
        new{T}(R)
    end
end

function f(x::T, 𝕊::Sphere) where {T<:AbstractArray}
    x[1]^2+x[2]^2+x[3]^2-𝕊.R^2
end

# Projection matrix
function P(x::T, 𝕊::Sphere) where {T<:AbstractArray}
    R, x, y, z = 𝕊.R, x[1], x[2], x[3]
    return [R^2-x^2 -x*y -x*z ; -x*y R^2-y^2 -y*z ; -x*z -y*z R^2-z^2]
end

"""
    Settings for the Torus 𝕋².
"""

struct Torus{T<:Real} <: EmbeddedManifold
    R::T
    r::T

    function Torus(R::T, r::T) where {T<:Real}
        if R<r
            error("R must be larger than or equal to r")
        end
        new{T}(R,r)
    end
end

function f(x::T, 𝕋::Torus) where {T<:AbstractArray}
    R, r, x, y, z = 𝕋.R, 𝕋.r, x[1], x[2], x[3]
    (x^2 + y^2 + z^2 + R^2 - r^2)^2 - 4.0*R^2*(x^2 + y^2)
end

# Projection matrix
function P(x::T, 𝕋::Torus) where {T<:AbstractArray}
    R, r, x, y, z = 𝕋.R, 𝕋.r, x[1], x[2], x[3]
    ∇f = [  4*x*(x^2+y^2+z^2+R^2-r^2) - 8*R^2*x,
            4*y*(x^2+y^2+z^2+R^2-r^2) - 8*R^2*y,
            4*z*(x^2+y^2+z^2+R^2-r^2)]# ForwardDiff.gradient((y)->f(y, 𝕋), x)
    n = ∇f./norm(∇f)
    return Matrix(Diagonal(ones(3))) .- n*n'
end


"""
    Settings for the Paraboloid ℙ²
"""

struct Paraboloid{T<:Real} <: EmbeddedManifold
    a::T
    b::T

    function Paraboloid(a::T, b::T) where {T<:Real}
        if a == 0 || b == 0
            error("parameters cannot be 0")
        end
        new{T}(a, b)
    end
end

function f(x::T, ℙ::Paraboloid) where {T<:AbstractArray}
    a, b, x, y, z = ℙ.a, ℙ.b, x[1], x[2], x[3]
    return (x/a)^2 + (y/b)^2 - z
end

function P(x::T, ℙ::Paraboloid) where {T<:AbstractArray}
    a, b, x, y, z = ℙ.a, ℙ.b, x[1], x[2], x[3]
    ∇f = [2*x/a, 2*y/b , -1]
    n = ∇f./norm(∇f)
    return Matrix(Diagonal(ones(d))) .- n*n'
end
