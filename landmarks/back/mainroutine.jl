# convert dual to float, while retaining float if type is float
deepvalue(x::Float64) = x
deepvalue(x::ForwardDiff.Dual) = ForwardDiff.value(x)
deepvalue(x) = deepvalue.(x)

"""
    compute guided proposal and loglikelihood,
    x0: starting point
    W: Wiener increment
    X: samplePath to write into
    ll0: logical flag whether to also compute rhotilde
    skip: nr of terms skipped in evaluating left-hand Riemann approximation to loglikelihood

    if x0 consists of Duals instead of floats, this is retained in computing the loglikelihood
    The elements of the simulated path are converted to floats
"""
function simguided_llikelihood!(::LeftRule, x0, W, Q, X; skip = 0, ll0 = true)
    Elx0 = eltype(x0)
    tt =  X.tt
    X.yy[1] .= deepvalue(x0)
    x = copy(x0)
    som::Elx0  = 0.
    for i in 1:length(tt)-1
        # compute bout en wout (drift increment and Wiener increment)
            if i<=length(tt)-1-skip         # likelihood terms
                # compute loglikehood contribution and increment som by its value
            end
            x .= x + dt * bout +  wout
            X.yy[i+1] .= deepvalue(x)
    end
    if ll0
        # compute logrhotilde at (0,x0) and save into logρ0
    else
        logρ0 = 0.0 # don't compute
    end
    som + logρ0
end

slogρ(W, Q, X) = (x) -> simguided_llikelihood!(LeftRule(), x, W, Q, X)

∇x = copy(x)

# following returns the gradient, for starting point x and writes it into ∇x
ForwardDiff.gradient!(∇x, slogρ(W, Q, X),x)


#########

SRC_DIR = joinpath(Base.source_dir(), "..", "src")
AUX_DIR = joinpath(SRC_DIR, "auxiliary")
OUT_DIR = joinpath(Base.source_dir(), "..", "output")
mkpath(OUT_DIR)

#include(joinpath(SRC_DIR, "BridgeSDEInference.jl"))
using BridgeSDEInference
using Distributions # to define priors
using Random        # to seed the random number generator
using DataFrames
using CSV

fptObsFlag = false

# pick dataset
filename = "path_part_obs_conj.csv"


param = :complexConjug
θ₀ = [10.0, -8.0, 15.0, 0.0, 3.0]
# Target law
P˟ = FitzhughDiffusion(param, θ₀...)

param = :simpleConjug
P = FitzhughDiffusion(param, 10.0, -8.0, 15.0, 0.0, 3.0)

#The starting point needs to be set and transformed to appropriate parametrisation :
x0 = ℝ{2}(-0.5, 0.6) # in regular parametrisation
x0 = regularToConjug(x0, P.ϵ, 0.0) # translate to conjugate parametrisation

dt = 1/50000
T = 10.0
tt = 0.0:dt:T


Random.seed!(4)
XX, _ = simulateSegment(0.0, x0, P, tt)

num_obs = 100
skip = div(length(tt), num_obs)
Time = collect(tt)[1:skip:end]
df = DataFrame(time=Time, x1=[x[1] for x in XX.yy[1:skip:end]],
               x2=[(i==1 ? x0[2] : NaN) for (i,t) in enumerate(Time)])

CSV.write(FILENAME_OUT, df)


# fetch the data
(df, x0, obs, obsTime, fpt,
  fptOrPartObs) = BridgeSDEInference.readData(Val(fptObsFlag), joinpath(OUT_DIR, filename))


  # Auxiliary law
P̃ = [FitzhughDiffusionAux(param, θ₀..., t₀, u[1], T, v[1]) for (t₀,T,u,v)
     in zip(obsTime[1:end-1], obsTime[2:end], obs[1:end-1], obs[2:end])]
display(P̃[1])






######## adaptation of Marcin's code

function solveAndll!(::EulerMaruyama, x0::T, W::SamplePath, P::GuidPropBridge, θ
                     ) where T
    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    tt =  Y.tt
    Y.yy[1] .= deepvalue(x0)
    y = copy(x0)
    ll::T  = 0.

    ww = W.yy
    tt = Y.tt
    yy = Y.yy
    tt[:] = W.tt

    ll = 0.0
    for i in 1:N-1
        yy[.., i] = y
        dWt = ww[.., i+1]-ww[.., i]
        s = tt[i]
        dt = tt[i+1]-tt[i]
        b_prop = _b((i,s), y, P, θ)
        y = y + b_prop*dt + _scale(dWt, σ(s, y, P, θ))

        b_trgt = _b((i,s), y, target(P), θ)
        b_aux = _b((i,s), y, auxiliary(P), θ)
        rₜₓ = r((i,s), y, P, θ)
        ll += dot(b_trgt-b_aux, rₜₓ) * dt

        if !constdiff(P)
            Hₜₓ = H((i,s), y, P, θ)
            aₜₓ = a((i,s), y, target(P), θ)
            ãₜ = ã((i,s), y, P, θ)
            ll -= 0.5*sum( (aₜₓ - ãₜ).*Hₜₓ ) * dt
            ll += 0.5*( rₜₓ'*(aₜₓ - ãₜ)*rₜₓ ) * dt
        end
        Y.yy[i+1] .= deepvalue(y)
    end
    yy[.., N] = endpoint(y, P)
    ll # here the log of rhotilde(0,x0) should be added
end

Elx0 = eltype(x0)
tt =  X.tt
X.yy[1] .= deepvalue(x0)
x = copy(x0)
som::Elx0  = 0.
for i in 1:length(tt)-1
    # compute bout en wout (drift increment and Wiener increment)
        if i<=length(tt)-1-skip         # likelihood terms
            # compute loglikehood contribution and increment som by its value
        end
        x .= x + dt * bout +  wout
        X.yy[i+1] .= deepvalue(x)
end
if ll0
    # compute logrhotilde at (0,x0) and save into logρ0
else
    logρ0 = 0.0 # don't compute
end
som + logρ0
end
