struct Wiener{T} <: ContinuousTimeProcess{T}
end
Wiener() = Wiener{Float64}()


struct WienerBridge{T} <: ContinuousTimeProcess{T}
    t::Float64  # end time
    v::T        # end point
end

function sample(tt, P::Wiener{T}) where T
    tt = collect(tt)
    yy = zeros(T,length(tt))
    sample!(SamplePath{T}(tt, yy), P)
end

function sample(tt, P::Wiener{T}, y1) where T
    tt = collect(tt)
    yy = zeros(T,length(tt))
    sample!(SamplePath{T}(tt, yy), P, y1)
end


function sample!(W::SamplePath{SVector{d,T}}, P::Wiener{SVector{d,T}}, y1 = zero(SVector{d,T})) where {d,T}
    sz = d
    W.yy[1] = y1
    yy = mat(W.yy)
    for i = 2:length(W.tt)
        rootdt = sqrt(W.tt[i]-W.tt[i-1])
        for j = 1:sz
            yy[sz*(i-1) + j] = yy[sz*(i-2) + j] + rootdt*randn(T)
        end
    end
    W
end

function sample!(W::VSamplePath{T}, P::Wiener{T}) where {T}
    N = length(W.tt)
    yy = W.yy
    sz = size(yy, 1)
    for i = 2:N
        rootdt = sqrt(W.tt[i] - W.tt[i-1])
        for j = 1:sz
            yy[j, i] = yy[j, i-1] + rootdt*randn(T)
        end
    end
    W
end

function sample!(W::SamplePath{T}, P::Wiener{T}, y1 = W.yy[1]) where T
    W.yy[1] = y1
    yy = W.yy
    for i = 2:length(W.tt)
        rootdt = sqrt(W.tt[i]-W.tt[i-1])
        yy[i] = yy[i-1] + rootdt*randn(T)
    end
    W
end

function sample(tt, P::WienerBridge{T}) where T
    tt = collect(tt)
    yy = zeros(T,length(tt))
    sample!(SamplePath{T}(tt, yy), P)
end

function sample(tt, P::WienerBridge{T}, y1) where T
    tt = collect(tt)
    yy = zeros(T,length(tt))
    sample!(SamplePath{T}(tt, yy), P, y1)
end

function sample!(W::SamplePath{SVector{d,T}}, P::WienerBridge{SVector{d,T}}, y1 = W.yy[1]) where {d,T}

    TT = P.t - W.tt[1]
    sz = d
    W.yy[1] = y1
    v = SVector(P.v)
    yy = mat(W.yy)

    wtotal = zeros(sz)
    for i = 2:length(W.tt)
        rootdt = sqrt(W.tt[i]-W.tt[i-1])
        for j = 1:sz
            wtotal[j] += yy[sz*(i-1) + j] = rootdt*randn(T)
        end
    end

    # noise between tt[end] and P.t
    rootdt = sqrt(P.t-W.tt[end])
    for j = 1:sz
            wtotal[j] += rootdt*randn(T) + (yy[j] - v[j])
    end

    # cumsum
    for i = 2:length(W.tt)
        dt = (W.tt[i]-W.tt[i-1])/TT
        for j = 1:sz
            yy[sz*(i-1) + j] = yy[sz*(i-2) + j] + yy[sz*(i-1) + j] - wtotal[j]*dt
        end
    end
    if W.tt[end] == P.t
        yy[:,end] = v
    end

    W
end

function sample!(W::SamplePath{T}, P::WienerBridge{T}, y1 = W.yy[1]) where T

    TT = P.t - W.tt[1]
    W.yy[1] = y1
    v = P.v
    yy = W.yy

    wtotal = zero(T)
    for i = 2:length(W.tt)
        rootdt = sqrt(W.tt[i]-W.tt[i-1])
        wtotal  += yy[i] = rootdt*randn(T)
    end

    # noise between tt[end] and P.t
    rootdt = sqrt(P.t-W.tt[end])
    wtotal += rootdt*randn(T)

    # normalize
    wtotal += (yy[1] - v)

    # cumsum
    for i = 2:length(W.tt)
        dt = (W.tt[i]-W.tt[i-1])/TT
        yy[i] = yy[i] + yy[i-1] - wtotal*dt
    end
    if W.tt[end] == P.t
        yy[end] = v
    end


    W
end

## drift and dispersion coefficients

function b(s, x, P::Wiener{T}) where T
    zero(T)
end

function bderiv(s, x, P::Wiener{T}) where T
    zero(outertype(P))
end

function σ(s, x, P::Wiener)
    I
end
function a(s, x, P::Wiener)
    I
end
function Γ(s, x, P::Wiener)
    I
end

function b!(s, x, dx, P::Wiener{T}) where T
    @. dx = 0
end

function σ!(s, x, dw, dm, P::Wiener)
    @. dm = dw
end





# transition density
transitionprob(s, x, t, P::Wiener{Float64}) = Normal(x,sqrt(t-s))
increment(dt, P::Wiener{Float64}) = Normal(0.0, sqrt(dt))

transitionprob(s, x, t, P::Wiener{SVector{N,Float64}}) where {N} = MvNormal(Vector(x),(t-s))
lp(s, x, t, y, P::Wiener{SVector{N,Float64}}) where {N} = logpdf(transitionprob(s, x, t, P), Vector(y))


function b(s, x, P::WienerBridge)
    (P.v-x)/(P.t-s)
end

function σ(s, x, P::WienerBridge)
    I
end

function a(s, x, P::WienerBridge)
    I
end
function Γ(s, x, P::WienerBridge)
    I
end
