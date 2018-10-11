# reminder, to type H*, do H\^+
cd("/Users/Frank/.julia/dev/Bridge")
outdir="/Users/Frank/Sync/DOCUMENTS/onderzoek/code/diffbridges/out_nclar/"

using Bridge, StaticArrays, Distributions
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models
using DelimitedFiles
using DataFrames
using CSV

T = 1.0
dt = 1/5000
τ(T) = (x) ->  x * (2-x/T)
tt = τ(T).(0.:dt:T)

sk = 0 # skipped in evaluating loglikelihood

νHparam = false
generate_data = true

# settings in case of νH - parametrisation
ϵ = 10^(-3)
Σdiagel = 10^(-10)

# settings sampler
iterations = 5*10^4
skip_it = 1000
subsamples = 0:skip_it:iterations

ρ = obs_scheme=="full" ? 0.85 : 0.95

########
#########
n = 3 # number of particles


# q are (p)ositions, p are (m)oments ;-)
qs = 1:2:2n
ps = 2:2:2n

q(i) = 2i - 1
p(i) = 2i


# Gaussian kernel
kernel(x, P) = 1/(2*π*P.a)^(length(x)/2)*exp(-norm(x)^2/(2*P.a))

const Point = SArray{Tuple{2},Float64,1,2}       # point in 2
const Unc = SArray{Tuple{2,2},Float64,2,4}     # Matrix presenting uncertainty
const NP = SVector{16,Float64}

symmetrize!(H) = H .= (H+H')/2

fll(B::Matrix{Unc}) = [B[(i+1)÷2,(j+1)÷2][mod1(i,2),mod1(j,2)] for i in 1:4n, j in 1:4n]
fll(v::Vector{Point}, n = n) = [v[(i+1)÷2][mod1(i,2)] for i in 1:4n]
sps(v::Vector) = [Point(v[i], v[i+1]) for i in 1:2:4n-1]
sps(B::Matrix) = [Unc(B[i, j], B[i+1, j], B[i, j+1], B[i+1,j+1] ) for i in 1:2:4n-1, j in 1:2:4n-1]




#########
#######

m, d = size(L)
Σ = SMatrix{m,m}(Σdiagel*I)

struct Landmarks <: ContinuousTimeProcess{Point}
    a::Float64 # kernel parameter
    σ::Float64 # noise level
    λ::Float64 # mean reversion
    n::Int
end

# specify auxiliary process
struct LandmarksTilde <: ContinuousTimeProcess{Point}
    a::Float64 # kernel parameter
    σ::Float64 # noise level
    λ::Float64 # mean reversion
    qT::Vector{Point}
    n::Int
end


aa = 10.0 ; λ = 0.5 ; si = 2.0

P = Landmarks(aa, si, λ, n)

function Bridge.b!(t, x, out, P::Landmarks)
    @. out = zero(Point)
    for i in 1:P.n
        for j in 1:P.n
            out[q(i)] += 0.5*x[p(j)]*kernel(x[q(i)] - x[q(j)], P)
            # heath bath
            out[p(i)] += -P.λ*0.5*x[p(j)]*kernel(x[q(i)] - x[q(j)], P) +
                1/(2*P.a) * dot(x[p(i)], x[p(j)]) * (x[q(i)]-x[q(j)])*kernel(x[q(i)] - x[q(j)], P)
        end
    end
    fll(out)
end
Bridge.b(t, x, P::Union{Landmarks, LandmarksTilde}) = Bridge.b!(t, x, zeros(Point, 2*P.n), P)

Bridge.b(1.0,rand(n*2),P)

Bridge.σ(t, x, P::NclarDiffusion) = ℝ{3}(0.0, 0.0, P.σ)
Bridge.constdiff(::Landmarks) = true






function Bridge._b!((i,s), x, out, Po::LandmarksBridge)
    Bridge.b!(Po.tt[i], x, out, Bridge.P(Po)) # original drift
    out2 = zero(out)
    Bridge.ri!(i, x, out2, Po)
    for i in 1:npoints(P) #acts only on p coordinates
        out[p(i)] += Bridge.P(Po).σ^2 * out2[p(i)]
    end
    out
end

Bridge.bitilde!(i, x, out, Po::LandmarksBridge) = Bridge.b!(Po.tt[i], x, out, Bridge.Pt(Po))

function Bridge.β(t, P::LandmarksTilde)
    zeros(Point, 2npoints(P))
end
function Bridge.B(t, P::LandmarksTilde)
    out = zeros(Unc, 2npoints(P), 2npoints(P))
    for i in 1:npoints(P)
        for j in 1:npoints(P)
            out[q(i),p(j)] += 0.5*one(Unc)*kernel(P.qT[i] - P.qT[j], P)
            out[p(i),p(j)] += -0.5*one(Unc)*P.λ*kernel(P.qT[i] - P.qT[j], P)
        end
    end
    out
end


Bridge.σ(t, x, P::NclarDiffusionAux) = ℝ{3}(0.0,0.0, P.σ)
Bridge.constdiff(::LandmarksTilde) = true
Bridge.b(t, x, P::NclarDiffusionAux) = Bridge.B(t,P) * x + Bridge.β(t,P)
Bridge.a(t, P::NclarDiffusionAux) = Bridge.σ(t,0,P) * Bridge.σ(t, 0, P)'

q0 = x0[qs] .+ sqrt(Σ)*randn(Point, n)
qT = X.yy[qs, end] .+ sqrt(Σ)*randn(Point, n)
pT = X.yy[2:2:end, end]

Pt = LandmarksTilde(aa, si, λ, qT, n)


# Solve Backward Recursion
Po = νHparam ? Bridge.PartialBridgeνH(tt, P, Pt, L, ℝ{m}(v),ϵ, Σ) : Bridge.PartialBridge(tt, P, Pt, L, ℝ{m}(v), Σ)

####################### MH algorithm ###################
W = sample(tt, Wiener())
X = solve(Euler(), x0, W, P)
Xo = copy(X)
solve!(Euler(),Xo, x0, W, Po)

solve!(Euler(),X, x0, W, Po)
ll = llikelihood(Bridge.LeftRule(), X, Po,skip=sk)

# further initialisation
Wo = copy(W)
W2 = copy(W)
XX = Any[]
if 0 in subsamples
    push!(XX, copy(X))
end

acc = 0

for iter in 1:iterations
    # Proposal
    sample!(W2, Wiener())
    #ρ = rand(Uniform(0.95,1.0))
    Wo.yy .= ρ*W.yy + sqrt(1-ρ^2)*W2.yy
    solve!(Euler(),Xo, x0, Wo, Po)

    llo = llikelihood(Bridge.LeftRule(), Xo, Po,skip=sk)
    print("ll $ll $llo, diff_ll: ",round(llo-ll,3))

    if log(rand()) <= llo - ll
        X.yy .= Xo.yy
        W.yy .= Wo.yy
        ll = llo
        print("✓")
        acc +=1
    end
    println()
    if iter in subsamples
        push!(XX, copy(X))
    end
end

@info "Done."*"\x7"^6

# write mcmc iterates to csv file

fn = outdir*"iterates-"*obs_scheme*".csv"
f = open(fn,"w")
head = "iteration, time, component, value \n"
write(f, head)
iterates = [Any[s, tt[j], d, XX[i].yy[j][d]] for d in 1:3, j in 1:length(X), (i,s) in enumerate(subsamples) ][:]
writecsv(f,iterates)
close(f)

ave_acc_perc = 100*round(acc/iterations,2)

# write info to txt file
fn = outdir*"info-"*obs_scheme*".txt"
f = open(fn,"w")
write(f, "Choice of observation schemes: ",obs_scheme,"\n")
write(f, "Easy conditioning (means going up to 1 for the rough component instead of 2): ",string(easy_conditioning),"\n")
write(f, "Number of iterations: ",string(iterations),"\n")
write(f, "Skip every ",string(skip_it)," iterations, when saving to csv","\n\n")
write(f, "Starting point: ",string(x0),"\n")
write(f, "End time T: ", string(T),"\n")
write(f, "Endpoint v: ",string(v),"\n")
write(f, "Noise Sigma: ",string(Σ),"\n")
write(f, "L: ",string(L),"\n\n")
write(f,"Mesh width: ",string(dt),"\n")
write(f, "rho (Crank-Nicholsen parameter: ",string(ρ),"\n")
write(f, "Average acceptance percentage: ",string(ave_acc_perc),"\n\n")
write(f, "Backward type parametrisation in terms of nu and H? ",string(νHparam),"\n")
close(f)


println("Average acceptance percentage: ",ave_acc_perc,"\n")
println(obs_scheme)
println("Parametrisation of nu and H? ", νHparam)
