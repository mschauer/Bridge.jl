import Base: dot, randn, rand
export supnorm
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

dot(J::Base.LinAlg.UniformScaling{Float64}, b::Float64) = J.λ*b
dot(b::Float64, J::Base.LinAlg.UniformScaling{Float64}) = J.λ*b
dot(x::Float64, y::Float64) = x*y


randn(::Type{Float64}) = randn()
randn{T}(::Type{Complex{T}}) = Complex(randn(T), randn(T))

