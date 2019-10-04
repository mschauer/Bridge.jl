using ForwardDiff

# convert dual to float, while retaining float if type is float
deepvalue(x::Float64) = x
deepvalue(x::ForwardDiff.Dual) = ForwardDiff.value(x)
deepvalue(x) = deepvalue.(x)

# here is a slight adaptation of Marcin's function solveAndll!
# should be able to get gradient information for starting point; new guided proposal is written into X
function solveAndll!(::EulerMaruyama, x0::T, W::SamplePath, X, P::GuidPropBridge, θ) where T
    N = length(W)
    N != length(X) && error("Y and W differ in length.")

    tt =  Y.tt
    X.yy[1] .= deepvalue(x0)
    y = copy(x0)
    ll::T  = 0.
    ww = W.yy

    ll::T = 0.0
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
        X.yy[i+1] .= deepvalue(y)
    end
    copyto!(X.yy[end], Bridge.endpoint(X.yy[end],P))
    ll # here the log of rhotilde(0,x0) should be added to ll, Marcin probabably wrote a function for that (please check it is Dual-number 'proof')
end



# apply as follows:
slogρ(W, X, P, θ) = (x0) -> solveAndll!(LeftRule(), 0x, W, X, P, θ)
∇x0 = copy(x0)
# following returns the gradient, for starting point x and writes it into ∇x
ForwardDiff.gradient!(∇x0, slogρ(W, X, P, θ),x0)
