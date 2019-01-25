
const Point = SArray{Tuple{d},Float64,1,d}       # point in R2
const Unc = SArray{Tuple{d,d},Float64,d,d*d}     # Matrix presenting uncertainty

abstract type UncMat end

struct State{P}
    q::Vector{P}
    p::Vector{P}
end

q(x::State) = x.q
p(x::State) = x.p
q(x::State, i) = x.q[i]
p(x::State, i) = x.p[i]

q(i::Int) = 2i - 1
p(i::Int) = 2i


State(xflat) = State(xflat[1:2:end], xflat[2:2:end])

import Base:iterate, eltype, copy, copyto!, zero, eachindex, getindex, setindex!, size, vec
iterate(x::State) = (x.q, true)
iterate(x::State, s) = s ? (x.p, false) : nothing
copy(x::State) = State(copy(x.q), copy(x.p))
function copyto!(x::State, y::State)
    copyto!(x.q, y.q)
    copyto!(x.p, y.p)
    x
end
function copyto!(x::State, y)
    for i in eachindex(x)
        x[i] = y[i]
    end
    x
end
#eltype(::State{P}) = P


Base.broadcastable(x::State) = x
Broadcast.BroadcastStyle(::Type{<:State}) = Broadcast.Style{State}()



size(s::State) = (2, length(s.p))

zero(x::State) = State(zero(x.p), zero(x.q))
zero!(v) = fill!(v, zero(eltype(v)))
zero!(v) = v[:] = fill!(v, zero(eltype(v)))

function zero!(x::State)
    zero!(x.p)
    zero!(x.q)
    x
end

function getindex(x::State, I::CartesianIndex)
    if I[1] == 1
        x.q[I[2]]
    elseif I[1] == 2
        x.p[I[2]]
    else
        throw(BoundsError())
    end
end
function setindex!(x::State, val, I::CartesianIndex)
    if I[1] == 1
        x.q[I[2]] = val
    elseif I[1] == 2
        x.p[I[2]] = val
    else
        throw(BoundsError())
    end
end

eachindex(x::State) = CartesianIndices((Base.OneTo(2), eachindex(x.p)))

import Base: *, +, /, -
import LinearAlgebra: norm
import Bridge: outer
*(c::Number, x::State) = State(c*x.q, c*x.p)
*(x::State,c::Number) = State(x.q*c, x.p*c)
+(x::State, y::State) = State(x.q + y.q, x.p + y.p)
-(x::State, y::State) = State(x.q - y.q, x.p - y.p)
/(x::State, y) = State(x.q/y, x.p/y)

vec(x::State) = vec([x[J] for J in eachindex(x)]) #
deepvec(x::State{P}) where {P} = vec([x[J][K] for K in eachindex(x.p[1]), J in eachindex(x)])


"""
Convert vector to State. it is assumed that x is ordered as follows
- position landmark 1
- momentum landmark 1
- position landmark 2
- momentum landmark 2
...
"""
function deepvec2state(x)
   m = div(length(x),2d)
   q = Vector{Point}(undef,m)
   p = Vector{Point}(undef,m)
   for i in 1:m
       q[i] = Point(x[(i-1)*2d .+ (1:d)])
       p[i] = Point(x[((i-1)*2d+d+1):(2i*d)])
   end
   State(q,p)
end

function deepmat(H::AbstractMatrix{S}) where {S}
    d1, d2 = size(S)
    reshape([H[i,j][k,l] for k in 1:d1, i in 1:size(H, 1), l in 1:d2, j in 1:size(H,2)], d1*size(H,1), d2*size(H,2))
end
#@test  outer(deepvec(x0)) == deepmat(outer(vec(x0)))



function deepmat2unc(A::Matrix)  # d is the dimension of the square subblocks
  k =size(A,1)
  m = div(k,d)
  [Unc(A[(i-1)*d+1:i*d,(j-1)*d+1:j*d]) for i in 1:m, j in 1:m]
end





function outer(x::State, y::State)
    [outer(x[i],y[j]) for i in eachindex(x), j in eachindex(y)]
end
# unc(a::Array) = State([sqrt(a[1, i, 1, i]) for i in 1:size(a, 2)],[sqrt(a[2, i, 2, i]) for i in 1:size(a, 2)])

# Frank: this seems wrong
norm(x::State) = norm(x.q) + norm(x.q)

function Base.show(io::IO, state::State)
  show(io,  "text/plain", hcat(q(state),p(state)))
end

struct InverseCholesky{T}
    L::T
end



"""
Solve L L'y =x using two backsolves,
L should be lower triangular
"""
function cholinverse!(L, x)
    LinearAlgebra.naivesub!(L, x) # triangular backsolves
    LinearAlgebra.naivesub!(UpperTriangular(L'), x)
    x
end

"""
Compute y = H*x where Hinv = L*L' (Cholesky decomposition

Input are L and x, output is y

y=Hx is equivalent to LL'y=x, which can be solved
by first backsolving Lz=x for z and next
backsolving L'y=z

L is a lower triangular matrix with element of type UncMat
x is a State or vector of points
Returns a State (Todo: split up again and return vector for vector input)
"""
function Base.:*(H::InverseCholesky, x::Union{State,Vector{Point}})
    y = copy(x)
    if isa(x,State)
        y = cholinverse!(H.L,  vec(x)) # triangular backsolves
    else
        y = cholinverse!(H.L,  x) # triangular backsolve
    end
    State(y[1:d:end], y[d:d:end])
end


"""
Compute y = H*X where Hinv = L*L' (Cholesky decomposition

Input are L and X, output is Y

y=Hx is equivalent to LL'y=X, which can be solved
by first backsolving LZ=X for z and next
backsolving L'Y=Z

L is a lower triangular matrix with element of type UncMat
X is a matrix with elements of type UncMat
Returns a matrix with elements of type UncMat
"""
function Base.:*(H::InverseCholesky, X)
    cholinverse!(H.L,  copy(X)) # triangular backsolves
end



# """
# Input:  H = inverse(L L'), with L lower-triangular
# Output: H*v
# """
# function Base.:*(H::InverseCholesky, v)
#     # two backsolves
#     y =copy(v)
#     cholinverse!(H.L,y)
#     y
# end




# L = LowerTriangular(rand(5,5))
# H = InverseCholesky(L)
# v = rand(5)
# H*v - inv(L*L')*v
#

lchol(A) = LowerTriangular(Matrix(LinearAlgebra._chol!(copy(A), UpperTriangular)[1])')


# di = 10
# L_ = Matrix(LowerTriangular(rand(di,di)))
# L = LowerTriangular(deepmat2unc(L_))
# H = InverseCholesky(L)
# v = reinterpret(Point, rand(di))
# H*v - deepmat2unc(inv(L_*L_'))*v

"""
Hdagger = U S U' with S = L L' (cholesky)
U has orthonormal columns, S is a small positive definite symmetric matrix given
in factorized form, e.g. `S <: Cholesky`

Then H*x is defined as U inv(S) U' (that is the Moore Penrose inverse)
"""
struct LowRank{TS,TU}
    S::TS
    U::TU
end

function Base.:*(H::LowRank, x::State)
    #z = H.S\(H.U' * deepvec(x)) # e.g. uses two triangular backsolves if S is `Cholesky`
    #deepvec2state(H.U*z)
    deepvec2state(H.U * (inv(H.S) * (H.U' * deepvec(x))))
end

# function isposdef(A::Array{StaticArrays.SArray{Tuple{2,2},Float64,2,4},2})
#     LinearAlgebra.isposdef(deepmat(A))
# end

if TEST
    A = reshape([Unc(1:4), Unc(5:8), Unc(9:12), Unc(13:16)],2,2)
    B = reshape([-Unc(1:4), Unc(3:6), Unc(9:12), Unc(13:16)],2,2)
    @test deepmat(A*B)==deepmat(A)*deepmat(B)
    @test deepmat(A') == deepmat(A)'
    @test deepmat(A*A')==deepmat(A)*deepmat(A')
    Q = deepmat2unc(Matrix(qr(deepmat(A)).Q))
    C = Q * Q'
    lchol(C)
    @test deepmat(lchol(C))== cholesky(deepmat(C)).L
    isposdef(deepmat(C))

    # check cholinverse!
    H = InverseCholesky(lchol(C))
    x = rand(4)
    xstate = deepvec2state(x)
    @test norm(deepvec(H*xstate) -  inv(deepmat(C))*x)<10^(-10)

    LinearAlgebra.naivesub!(lchol(C), A) # triangular backsolves
end
