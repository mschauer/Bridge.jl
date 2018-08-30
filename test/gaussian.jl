using Bridge
using Test, LinearAlgebra, Random
using Distributions, StaticArrays
using Bridge: Gaussian, PSD

let
    μ = rand()
    x = rand()
    Σ = rand()^2

    p = pdf(Normal(μ, √Σ), x)
    @test pdf(Gaussian(μ, Σ), x) ≈ p
    @test pdf(Gaussian(μ, Σ*I), x) ≈ p
    @test pdf(Gaussian([μ], [√Σ]*[√Σ]'), [x]) ≈ p

    @test pdf(Gaussian((@SVector [μ]), @SMatrix [Σ]), @SVector [x]) ≈ p
end

for d in 1: 3
    μ = rand(d)
    x = rand(d)
    rΣ = tril(rand(d,d))
    Σ = rΣ*rΣ'
    p = pdf(MvNormal(μ, Σ), x)

    @test pdf(Gaussian(μ, Σ), x) ≈ p
    @test pdf(Gaussian(μ, PSD(rΣ)), x) ≈ p
    @test pdf(Gaussian(SVector{d}(μ), SMatrix{d,d}(Σ)), x) ≈ p
    @test pdf(Gaussian(SVector{d}(μ), PSD(SMatrix{d,d}(rΣ))), x) ≈ p
end

for d in 1: 3
    μ = rand(d)
    x = rand(d)
    rΣ = rand()
    Σ = Matrix(I, d, d)*rΣ^2
    p = pdf(MvNormal(μ, Σ), x)

    @test pdf(Gaussian(μ, rΣ^2*I), x) ≈ p
    @test pdf(Gaussian(SVector{d}(μ), SDiagonal(rΣ^2*ones(SVector{d}))), x) ≈ p
    @test pdf(Gaussian(SVector{d}(μ), SMatrix{d,d}(Σ)), x) ≈ p
end