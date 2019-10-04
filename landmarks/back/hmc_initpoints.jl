struct DeterMSshadow2{T} <: ContinuousTimeProcess{State{PointF}}
    a::T # kernel std parameter
    c::T # kernel multiplicate parameter
    λ::T # mean reversion
    n::Int
    h::Float64 # stdev
    p0 # fixed true Hamiltonian momenta at time zero
    ∇ # direction, vector of Points
end

Pshadow = DeterMSshadow(0.1, 0.1, 0.0, P.n, h, x0.p , ∇xp)
DeterMSshadow2(0.1, 0.1, 0.0, P.n, 0.1, rand(PointF,4) , rand(PointF,2))

# note: here x = (q,pshadow)
function Bridge.b!(t, x, out, P::DeterMSshadow2)
    zero!(out)
    for i in 1:P.n
        for j in 1:P.n
            out.q[i] += ((x.p[j]-P.∇[j])/P.h + P.p0[j]) * kernel(q(x,i) - q(x,j), P)
            out.p[i] += -P.λ*P.p0[j]*kernel(q(x,i) - q(x,j), P) -
                 (dot(P.p0[i], P.p0[j]) + dot(x.p[i]-P.∇[i], x.p[j]-P.∇[j])/P.h) * ∇kernel(q(x,i) - q(x,j), P)
        end
    end
    out
end


"""
kernel in Hamiltonian
"""
function kernel(q, P::DeterMSshadow2)
 #(2*π*P.a^2)^(-d/2)*exp(-Bridge.inner(q)/(2*P.a^2))
 P.c * exp(-Bridge.inner(q)/(2*P.a^2))
end

"""
gradient of kernel in hamiltonian
"""
function ∇kernel(q, P::DeterMSshadow2)
    -P.c * P.a^(-2) * kernel(q, P) * q
end

function Bridge.σ!(t, x, dm, out, P::DeterMSshadow2)
    zero!(out.q)
    out.p .= dm*0
    out
end

function hamiltonian(x::NState, P::DeterMSshadow2)
    s = 0.0
    for i in 1:P.n, j in 1:P.n
        s += (dot(x.p[i], x.p[j]) + dot(P.p0[i]-P.∇[i],P.p0[j]-P.∇[j])/P.h )* kernel(x.q[i] - x.q[j], P)
    end
    0.5 * s
end
