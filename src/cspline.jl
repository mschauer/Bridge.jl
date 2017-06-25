using Compat, StaticArrays
function cspline(s, t1, t2, p1, p2, m1, m2)
    d = t2-t1
    t = (s-t1)/(t2-t1)
    t2 = t*t
    t3 = t2*t
    z = (@SMatrix [2. -3. 0. 1.; -2. 3. 0. 0.; 1. -2. 1. 0.; 1. -1. 0. 0.])* (@SVector [t3, t2, t, 1.])
    z[1]*p1 + z[2]*p2 + z[3]*d*m1 + z[4]*d*m2
end
function intcspline(s, t1, t2, p1, p2, m1, m2)
        d = t2-t1
        t = (s-t1)/(t2-t1)
        t2 = t*t
        t3 = t2*t
        t4 = t2*t2
        t4, t3, t2 = t4/4, t3/3, t2/2
        z = (@SMatrix [2. -3. 0. 1.; -2. 3. 0. 0.; 1. -2. 1. 0.; 1. -1. 0. 0.])* (@SVector [t4, t3, t2, t])
        (z[1]*p1 + z[2]*p2 + z[3]*d*m1 + z[4]*d*m2)*d
end
intcspline(s, T, t1, t2, p1, p2, m1, m2) = intcspline(T, t1, t2, p1, p2, m1, m2) - intcspline(s, t1, t2, p1, p2, m1, m2)

mutable struct CSpline{T}
    s; t; x::T; y::T; mx; my
end
CSpline{T}(s, t, x::T, y = x, m0 = (y-x)/(t-s), m1 =  (y-x)/(t-s)) = CSpline{T}(s, t, x, y, mx, my)
(cs::CSpline)(t) =  cspline(t, cs.s, cs.t, cs.x, cs.y, cs.mx, cs.my)
#call(cs::CSpline, t) =  cspline(t, cs.s, cs.t, cs.x, cs.y, cs.mx, cs.my)

integrate(cs::CSpline, s, t) = intcspline(s,t, cs.s, cs.t, cs.x, cs.y, cs.mx, cs.my)
