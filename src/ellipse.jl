import Base.getindex, Base.setindex!
const .. = Val{:...}

setindex!(A::AbstractArray{T,1}, x, ::Type{Val{:...}}, n) where {T} = A[n] = x
setindex!(A::AbstractArray{T,2}, x, ::Type{Val{:...}}, n) where {T} = A[ :, n] = x
setindex!(A::AbstractArray{T,3}, x, ::Type{Val{:...}}, n) where {T} = A[ :, :, n] = x

getindex(A::AbstractArray{T,1}, ::Type{Val{:...}}, n) where {T} = A[n]
getindex(A::AbstractArray{T,2}, ::Type{Val{:...}}, n) where {T} = A[ :, n]
getindex(A::AbstractArray{T,3}, ::Type{Val{:...}}, n) where {T} = A[ :, :, n]

export .., getindex, setindex!
