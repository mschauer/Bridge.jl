using Makie
using Distributions
using ForwardDiff
using Trajectories
using LinearAlgebra
using Colors


struct DoubleWell
end
struct DoubleWellTilde
end
P = DoubleWell()
P̃ = DoubleWellTilde()
π(x, ::DoubleWell) = exp(-(x+4)^2/2) + exp(-(x-4)^2/2)
π(x, ::DoubleWellTilde) = exp(-(x)^2/(10*2))
logϕ(p, P) = -inner(p)/2
logπ(x, P) = log(π(x, P))
Dlogπ(q, P) = ForwardDiff.derivative(x -> logπ(x, P), q)

inner(p) = dot(p, p)
H(q, p, P) = -log(π(q, P)) + inner(p)/2
DqH(q, p, P) = ForwardDiff.derivative(x -> H(x, p, P), q)
DpH(q, p, P) = ForwardDiff.derivative(x -> H(q, x, P), p)

function leapfrog_step(t, (q, p), tnew, P, σ)
    h = tnew - t

    #p = p - σ^2*p*h + σ*sqrt(2*h)*randn()
    ρ = (1-σ^2*h)
    p = ρ*p + sqrt(1-ρ^2)*randn()

    p = p - h*DqH(q, p, P)
    q = q + h*DpH(q, p, P)

    tnew, (q, p)
end

function hmc!(X, P, P̃)
    t = X.t[end]
    q, p = X.x[end]
    if P == P̃
        ρ = 0.75
        flip = 1
        h = 0.0
        σ = 0.0
    else
        ρ = 0.76
        flip = -1
        h = 0.3
        σ = 0.0
    end
    subsample = 10


    m = 100
    Δt = .5
    acc = 0


    n = 100_000
    for i in 1:n
        p = ρ*p + sqrt(1-ρ^2)*randn()

        tᵒ = t
        qᵒ, pᵒ = q, p
        Z = randn()
        #pᵒ = pᵒ + h*(Dlogπ(q, P) - 0pᵒ)
        pᵒ = pᵒ + sqrt(h)*Z
        tᵒ, (qᵒ, pᵒ) = leapfrog_step(tᵒ, (qᵒ, pᵒ), tᵒ + h, P, 0.0)
        h2 = (1 + 1rand())*Δt/m

        for i in 1:m
            tᵒ, (qᵒ, pᵒ) = leapfrog_step(tᵒ, (qᵒ, pᵒ), tᵒ + h2, P̃, σ)
        end
        tᵒ, (qᵒ, pᵒ) = leapfrog_step(tᵒ, (qᵒ, pᵒ), tᵒ + h, P, 0.0)
        pᵒ = pᵒ + sqrt(h)*Z
        pᵒ = flip*pᵒ
#        pᵒ = pᵒ - h*(Dlogπ(qᵒ, P) - 0pᵒ)


        if P ≠ P̃
            ll = H(q, p, P) - H(qᵒ, pᵒ, P)
            #ll += logϕ(-sqrt(h)*(Dlogπ(q, P) + Dlogπ(qᵒ, P)) - Z, P) - logϕ(Z, P)
            #ll += logϕ(sqrt(h)*Dlogπ(qᵒ, P) - Z2, P) - logϕ(Z2, P)
            #ll += logϕ(Z + 2sqrt(h)*Dlogπ(q, P), P) - logϕ(Z, P)

            if log(rand()) ≤ ll
                q, p = qᵒ, pᵒ
                acc += 1
            end
        else
            q, p = qᵒ, pᵒ
            acc += 1
        end
        p = flip*p # second flip
        t = tᵒ
        i % subsample == 0 && push!(X, t=>(q, p))
    end
    @show acc/n
end

splitup(x) = (first.(x), last.(x))
splitup(X::Trajectory) = (first.(X.x), last.(X.x))


t = 0.0
q, p = randn(), randn()
X = Trajectory([t],[(q,p)])
Y = Trajectory([t],[(q,p)])

hmc!(X, P, P̃)
hmc!(Y, P, P)
println("doubles ",  sum(first.(X.x)[1:end-1] .== first.(X.x)[2:end] )/length(X))
#scatter(X.x, markersize=0.04, color=RGBA(1,0,1,0.1))
#scatter!(Y.x, markersize=0.04, color=RGBA(1,1,0,0.1))

p1a = scatter(X.x, markersize=0.04)
lines!(X.x, linewidth=0.1)
p1b = scatter(Y.x, color=:blue, markersize=0.04)
lines!(Y.x, linewidth=0.08, color=:blue)

p1 = hbox(p1a,p1b)
p2 = hbox(lines(first.(X.x)), lines(first.(Y.x), color=:blue))
vbox(p1, p2)
