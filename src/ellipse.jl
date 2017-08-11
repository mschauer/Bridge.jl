import Base.getindex, Base.setindex!
const .. = Val{:...}

setindex!{T<:Any, Ti<:Integer}(A::AbstractArray{T,Ti}, x, ::Type{Val{:...}}, ::Base.Colon) = throw( ArgumentError("cannot combine ellipse with colon this way"))
getindex(::Base.SparseArrays.SparseMatrixCSC, ::Type{Val{:...}}, ::Base.Colon) = throw( ArgumentError("cannot combine ellipse with colon this way"))

setindex!{T}(A::AbstractArray{T,1}, x, ::Type{Val{:...}}, n) = A[n] = x
setindex!{T}(A::AbstractArray{T,2}, x, ::Type{Val{:...}}, n) = A[ :, n] = x
setindex!{T}(A::AbstractArray{T,3}, x, ::Type{Val{:...}}, n) = A[ :, :, n] =x

getindex{T}(A::AbstractArray{T,1}, ::Type{Val{:...}}, n) = A[n]
getindex{T}(A::AbstractArray{T,2}, ::Type{Val{:...}}, n) = A[ :, n]
getindex{T}(A::AbstractArray{T,3}, ::Type{Val{:...}}, n) = A[ :, :, n]

export .., getindex, setindex!
