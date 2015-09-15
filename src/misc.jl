function cumsum0(dx::Vector)
        n = length(dx) + 1
        x = similar(dx, n)
        x[1] = 0.0      
        for i in 2:n
                x[i] = x[i-1] + dx[i-1] 
        end
        x
end

import Base.Random: randn

randn(::Type{Float64}) = randn()
randn{T}(::Type{Complex{T}}) = Complex(randn(T), randn(T))
immutable RandnFunctor{T} <: FixedSizeArrays.Func{1} end
@inline call{T}(rf::Type{RandnFunctor{T}}, i...) = randn(T)
@inline randn{FSA <: FixedArray}(x::Type{FSA}) = map(RandnFunctor{eltype(FSA)}, FSA)