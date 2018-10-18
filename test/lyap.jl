
using Bridge, Random, Test, StaticArrays

struct Diff_Lyap{N} <: ContinuousTimeProcess{SVector{N,Float64}}
end
Diff_Lyap(d) = Diff_Lyap{d}()

dim(::Diff_Lyap{N}) where {N} = N
@testset "Lyap" begin
    # example setting
    d = 20
    t = 2.0:0.01:5.0

    Random.seed!(4)
    Bridge.B(t, P::Diff_Lyap) = reshape(rand(dim(P)^2),dim(P),dim(P))
    Bridge.σ(t, P::Diff_Lyap) = reshape(rand(10*dim(P)),dim(P),10)
    Bridge.a(t, P::Diff_Lyap) = Bridge.σ(t,P) * Bridge.σ(t,  P)'
    Pt = Diff_Lyap(d)

    Hend = Matrix(1.0I, d,d)
    out = Bridge.lyapunovpsdbackward(t, Pt, Hend)

    @test all(map(x-> isposdef(Matrix(Bridge.symmetrize(x))), out.yy))
end
