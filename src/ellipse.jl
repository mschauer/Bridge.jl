
  import Base.getindex, Base.setindex!
  const ..   = Val{:..}

  setindex!{T,N,N2}(A::AbstractArray{T,N}, x, ::Type{Val{:..}}, n::Vararg{Int,N2}) = A[(Colon() for i in 1:N-N2)..., n...] = x
  getindex{T,N,N2}(A::AbstractArray{T,N}, ::Type{Val{:..}}, n::Vararg{Int,N2}) = A[(Colon() for i in 1:N-N2)..., n...]

  setindex!{T,N}(A::AbstractArray{T,N}, x, n::Int, ::Type{Val{:..}}) = A[n, (Colon() for i in 1:N-1)...] = x
  #setindex!{T,N,N2}(A::AbstractArray{T,N}, x, n::Vararg{Int,N2}, ::Type{Val{:..}}) = A[n, (Colon() for i in 1:N-N2-1)...] = x

  getindex{T,N}(A::AbstractArray{T,N}, n::Int, ::Type{Val{:..}}) = A[n,(Colon() for i in 1:N-1)...]
  #getindex{T,N,N2}(A::AbstractArray{T,N}, n::Vararg{Int,N2}, ::Type{Val{:..}}) = A[n,(Colon() for i in 1:N-N2-1)...]

  export .., getindex, setindex!
