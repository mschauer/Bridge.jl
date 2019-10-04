

using Test
using StaticArrays
using LinearAlgebra

const d = 2
const Point = SArray{Tuple{d},Float64,1,d}
const Unc = SArray{Tuple{d,d},Float64,d,d*d}

x = [Point(rand(d)...) for i in 1:100]

A = I + x*x'
lchol(A) = LowerTriangular((LinearAlgebra._chol!(copy(A), UpperTriangular)[1])')

rchol(A) = LinearAlgebra._chol!(copy(A), UpperTriangular)[1]

L = lchol(A)
R = rchol(A)

@test norm(L*L' - A) < 1e-8

y = copy(x)
LinearAlgebra.naivesub!(L, y) # triangular backsolves
LinearAlgebra.naivesub!(R, y)

@test norm(A*y - x) < 1e-9
