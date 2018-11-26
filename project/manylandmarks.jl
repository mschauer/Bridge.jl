using Bridge
using Bridge: MeanCov, iterateall
using StaticArrays
using Base.Iterators, LinearAlgebra, Statistics, Random, DelimitedFiles
using BenchmarkTools
PLOT = :makie

kernel(x, a::Float64 = 1.2) = 2exp(-norm(x)^2/(2*a))


@dev Bridge
kernel(x, a::Float64 = 1.2) = 2exp(-norm(x)^2/(2*a))


struct Arg4Closure{B,T}
    f::B
    arg4::T
end
(f::Arg4Closure)(arg1, arg2, arg3) = f.f(arg1, arg2, arg3, f.arg4)


struct State{P}
    q::Vector{P}
    p::Vector{P}
end
q(x::State) = x.q
p(x::State) = x.p
q(x::State, i) = x.q[i]
p(x::State, i) = x.p[i]

Point = Float64


import Base:iterate, copy, copyto!, zero, eachindex, getindex, setindex!, size
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

Base.broadcastable(x::State) = x
Broadcast.BroadcastStyle(::Type{<:State}) = Broadcast.Style{State}()



size(s::State) = (2, length(s.p))

zero(x::State) = State(zero(x.p), zero(x.q))


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
function outer(x::State, y::State)
    [outer(x[i],y[j]) for i in eachindex(x), j in eachindex(y)]
end

unc(a::Array) = State([sqrt(a[1, i, 1, i]) for i in 1:size(a, 2)],[sqrt(a[2, i, 2, i]) for i in 1:size(a, 2)])

norm(x::State) = norm(x.q) + norm(x.q)

/(x::State, y) = State(x.q/y, x.p/y)


"""
Landmarks process

Describes the evolution of `n` landmarks.
"""
struct Landmarks <: ContinuousTimeProcess{State{Point}}
    a::Float64 # kernel parameter
    σ::Float64 # noise level
    λ::Float64 # mean reversion
    n::Int
end
kernel(x, P::Landmarks) = kernel(x, P.a)/P.n
npoints(P::Landmarks) = P.n

struct LandmarksTilde{T,S} <: ContinuousTimeProcess{State{Point}}
    a::Float64 # kernel parameter
    σ::Float64 # noise level
    λ::Float64 # mean reversion
    n::Int
    K::T
    p::S
end

LandmarksTilde(P, xT::State) =
LandmarksTilde(P.a, P.σ, P.λ, P.n, [kernel(x - y, P) for x in q(xT), y in q(xT)], p(xT))

npoints(P::LandmarksTilde) = P.n



zero!(v) = v[:] = fill!(v, zero(eltype(v)))
function hamiltonian((q, p), P)
    s = 0.0
    for i in eachindex(q), j in eachindex(q)
      #  i == j && continue
        s += 1/2*dot(p[i], p[j])*kernel(q[i] - q[j], P)
    end
    s
end

function Bridge.b!(t, (q, p), (qout, pout), P::Landmarks)
    zero!(qout)
    zero!(pout)
    for i in eachindex(q), j in eachindex(q)
           # i == j && continue
           # kernel(q[i] - q[j], P) > 1e-6 || continue
            K = kernel(q[i] - q[j], P)
            qout[i] += p[j]*K
            # heath bath
            pout[i] += -P.λ*p[j]*K + 
                1/(P.a) * dot(p[i], p[j]) * (q[i] - q[j])*K
    end
end

function Bridge.b!(t, (q, p), (qout, pout), P::LandmarksTilde)
    zero!(qout)
    zero!(pout)
    for i in eachindex(q), j in eachindex(q)
            qout[i] += p[j]*P.K[i,j]
            # heath bath
            pout[i] += -P.λ*p[j]*P.K[i,j] + 
                1/(P.a) * dot(P.p[i], P.p[j]) * (q[i] - q[j])*P.K[i,j]
    end
end
    
    
function Bridge.σ!(t, (q, p), dw, dm, P)
    (dmq, dmp) = dm
    zero!(dmq)
    for i in eachindex(dw)
        dmp[i] = P.σ*dw[i]
    end
    dm
end   

import Bridge: b!, σ!
function Bridge.solve!(::EulerMaruyama!, Y, u, W::SamplePath, P::Bridge.ProcessOrCoefficients) 
    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    tt = Y.tt
    tt[:] = W.tt
    yy = Y.yy
    y = copy(u)

    tmp1 = copy(y)
    tmp2 = copy(y)
    dw = copy(W.yy[1]) 
    for i in 1:N-1
        t¯ = tt[i]
        dt = tt[i+1] - t¯ 
        copyto!(yy[i], y)
        for k in eachindex(dw)
            dw[k] = W.yy[i+1][k] - W.yy[i][k]
        end
        b!(t¯, y, tmp1, P)
        σ!(t¯, y, dw, tmp2, P)
        for k in eachindex(y)
            y[k] = y[k] + tmp1[k]*dt + tmp2[k]
        end
    end
    copyto!(yy[end], y)
    Y
end


function Bridge.solvebackward!(::EulerMaruyama!, Y::SamplePath, u, W::SamplePath, P::Bridge.ProcessOrCoefficients) 
    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    tt = Y.tt
    tt[:] = W.tt
    yy = Y.yy
    y = copy(u)

    tmp1 = copy(y)
    tmp2 = copy(y)
    dw = copy(W.yy[1]) 
    for i in N:-1:2
        t¯ = tt[i]
        dt = tt[i-1] - t¯ 
        copyto!(yy[i], y)
        for k in eachindex(dw)
            dw[k] = W.yy[i-1][k] - W.yy[i][k]
        end
        b!(t¯, y, tmp1, P)
        σ!(t¯, y, dw, tmp2, P)
        for k in eachindex(y)
            y[k] = y[k] + tmp1[k]*dt + tmp2[k]
        end
    end
    copyto!(yy[1], y)
    Y
end

function Bridge.sample!(W::SamplePath{Vector{T}}, P::Wiener{Vector{T}}, y1 = W.yy[1]) where {T}
    y = copy(y1)
    copyto!(W.yy[1], y)
    
    for i = 2:length(W.tt)
        rootdt = sqrt(W.tt[i]-W.tt[i-1])
        for k in eachindex(y)
            y[k] =  y[k] + rootdt*randn(T)
        end
        copyto!(W.yy[i], y)
    end
    #println(W.yy[1])
    W
end


a = 0.6^2
#C = 1/(2*π*a)^(length(Point)/2)

λ = .0
σ = 1*0.01
n = 100

P = Landmarks(a, σ, λ, n)


q0 = sort(1*rand(n))
p0 = 0.1 .+ 0.1*randn(n)
x0 = State(q0, p0)
 

T = 2.0
t = 0.:0.001:T

Bridge.samplepath(tt, v) = SamplePath(tt, [copy(v) for t in tt]) 
X = Bridge.samplepath(t, zero(x0))
npilots = 20
Xb = [Bridge.samplepath(t, zero(x0)) for k in 1:npilots]
Xtb = [Bridge.samplepath(t, zero(x0)) for k in 1:npilots]

W = Bridge.samplepath(t, zero(q0))

Wb = Bridge.samplepath(t, zero(q0))

@time sample!(W, Wiener{Vector{Point}}())
@time sample!(Wb, Wiener{Vector{Point}}())


@time solve!(EulerMaruyama!(), X, x0, W, P)
xT = X.yy[end]
Pt = LandmarksTilde(P, xT)



@time for k in 1:npilots
    sample!(Wb, Wiener{Vector{Point}}())
    solvebackward!(EulerMaruyama!(), Xb[k], xT, Wb, P)
    solvebackward!(EulerMaruyama!(), Xtb[k], xT, Wb, Pt)
    
end
#Xbar = SamplePath(X.tt, [mean(Xb[k].yy[i] for k in 1:npilots) for i in 1:length(X.tt)])

z = [iterateall(MeanCov(Xb[k].yy[i] for k in 1:npilots)) for i in 1:length(X.tt)]
Xbar = SamplePath(X.tt, first.(z))
Xcov = SamplePath(X.tt, map(z -> z[2]*z[3], z))
Xunc = SamplePath(X.tt, map(unc, Xcov.yy))

z = [iterateall(MeanCov(Xtb[k].yy[i] for k in 1:npilots)) for i in 1:length(X.tt)]
Xtbar = SamplePath(X.tt, first.(z))
Xtcov = SamplePath(X.tt, map(z -> z[2]*z[3], z))
Xtunc = SamplePath(X.tt, map(unc, Xtcov.yy))

ss = 1:10:length(X.tt)


#println("...solve H")
#H⁺ = GSamplePath(tt, zeros(Unc, 2n, 2n, length(tt)))
V = Bridge.samplepath(t, zero(x0))
#f!(a, b, c, Pt) = BHHBta!(a, b, c, Pt)
#@time solvebackward!(Bridge.R3!(), F!Closure(BHHBta!, Pt), H⁺, H⁺T)

#println("...solve V")
solvebackward!(Bridge.R3!(), Arg4Closure(Bridge.b!, Pt), V, xT)





if @isdefined(PLOT) && PLOT == :makie
    using Makie
    include("../extra/makie.jl")
    function band!(scene, x, ylower, yupper; nargs...)
        n = length(x)
        coordinates = [x ylower; x yupper]
        ns = 1:n-1
        ns2 = n+1:2n-1
        connectivity = [ns ns .+ 1 ns2;
                        ns2 ns2 .+ 1 ns .+ 1]
        mesh!(scene, coordinates, connectivity; shading = false, nargs...)
    end
    c = Bridge.viridis(1:n);
end

if false
    c = Bridge.viridis(1:n);
    c2 = Bridge.viridis(1:n, 0.1f0)
    sc = Scene();  
    [lines!(sc, X.tt, q.(X.yy, i), linewidth=2.0, color=c[i]) for i in 1:n]; sc
    #[lines!(sc, Xbar.tt, q.(Xbar.yy, i) .+  q.(Xunc.yy, i), color=c[i]) for i in 1:n]; sc  
    #[lines!(sc, Xbar.tt, q.(Xbar.yy, i) .-  q.(Xunc.yy, i), color=c[i]) for i in 1:n]; sc  
    #[band!(sc, Xbar.tt, q.(Xbar.yy, i) .-  1.96*q.(Xunc.yy, i), q.(Xbar.yy, i) .+  1.96*q.(Xunc.yy, i), color=c2[i]) for i in 1:n]
    [band!(sc, Xtbar.tt, q.(Xtbar.yy, i) .-  1.96*q.(Xtunc.yy, i), q.(Xtbar.yy, i) .+  1.96*q.(Xtunc.yy, i), color=c2[i]) for i in 1:n]
    sc
    # [scatter!(sc, Xbar.tt[ss], q.(Xbar.yy[ss], i), color=c[i], markersize=q.(Xunc.yy[ss], i)) for i in 1:n]; sc
end

if false
  
   
    sc = Scene();  
    #[lines!(sc, X.tt, q.(Xtbar.yy, i), linewidth=2.0, color=c[i]) for i in 1:n]; sc
    [lines!(sc, X.tt, q.(V.yy, i), linewidth=2.0, color=c[i]) for i in 1:n];
    [band!(sc, Xtbar.tt, q.(Xtbar.yy, i) .-  1.96*q.(Xtunc.yy, i), q.(Xtbar.yy, i) .+  1.96*q.(Xtunc.yy, i), color=c2[i]) for i in 1:n];
    sc
end

#y = sample(t, Wiener()).yy
#lines(t, y)
#band!(Scene(), t, y-1, y+1, color = RGBA(1.0, 0, 0, 0.3))
