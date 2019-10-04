
using DualNumbers

# small test to understand
ff(x) = x^3
ff(Dual(2,1))

# Redefine
# const Point = SArray{Tuple{d},Float64,1,d}       # point in R2
# const Unc = SArray{Tuple{d,d},Float64,d,d*d}     # Matrix presenting uncertainty
# to

const Point{T} = SArray{Tuple{d},T,1,d}
const Unc{T} = SArray{Tuple{d,d},T,d,d*d}
# > Point{Dual{Float64}}(rand(3))

# matrix multiplication of mat of Uncs
function Base.:*(A::Array{Unc{T},2},B::Array{Unc{T},2}) where T
    C = zeros(Unc{T},size(A,1), size(B,2))
    for i in 1:size(A,1)
        for j in 1:size(B,2)
            for k in 1:size(A,2)
               C[i,j] += A[i,k] * B[k,j]
            end
        end
    end
    C
end

function Base.:*(A::Array{Unc,2},x::State)
    vecofpoints2state(A*vec(x))
end


# DON'T NEED THIS PROBABLY
#
# function ll(x0,XX,Q)
#     XX.yy[1] = deepvec2state(x0)
#     llikelihood(LeftRule(), XX, Q; skip = 0)
# end
# y0 = deepvec(x0)
# ll(y0,XX,Q)
#
# F(XX,Q) = (x) -> ll(x,  XX,Q)
# F(XX,Q)(y0)
# using ForwardDiff
#
# g = x0 -> ForwardDiff.gradient(F(XX,Q), x0) # g = ∇f
# g(deepvec(x0))
