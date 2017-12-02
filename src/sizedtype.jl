import Base: one, size, zero

struct SizedType{S,T}
    sz::S
    SizedType(::Type{T}, sz::S) where {S,T} = new{S,T}(sz)
end

"""
    sizedtype(x) -> T

Return an extended type which preserves `size` information. Makes `one(T)` and `zero(T)`
for vectors possible.
"""
sizedtype(x::T) where {T} =  SizedType{T}(())
sizedtype(x::Array{T}) where {T} = SizedType(T, size(x))
size(st::SizedType) = st.sz
basetype(st::SizedType{S,T}) where {S,T} = T

one(st::SizedType) = ones(basetype(st), size(st))
one(st::SizedType{Tuple{}}) = one(basetype(st))

zero(st::SizedType) = zeros(basetype(st), size(st))
zero(st::SizedType{Tuple{}}) = zero(basetype(st))