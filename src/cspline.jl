using StaticArrays
function cspline(s, t1, t2, p1, p2, m1, m2)
    d = t2-t1
    t = (s-t1)/(t2-t1)
    t2 = t*t
    t3 = t2*t
    z1, z2, z3, z4 = (1 - 3*t2 + 2*t3,  3*t2 - 2*t3,  t - 2*t2 + t3, 0 - t2 + t3)
    z1*p1 + z2*p2 + z3*d*m1 + z4*d*m2
end
function intcspline(s, t1, t2, p1, p2, m1, m2)
        d = t2-t1
        t = (s-t1)/(t2-t1)
        t2 = t*t
        t3 = t2*t
        t4 = t2*t2
        t4, t3, t2 = t4/4, t3/3, t2/2
        z1, z2, z3, z4 = (t - 3*t3 + 2*t4,  3*t3 - 2*t4,  t2 - 2*t3 + t4, 0 - t3 + t4)
        (z1*p1 + z2*p2 + z3*d*m1 + z4*d*m2)*d
end
intcspline(s, T, t1, t2, p1, p2, m1, m2) = intcspline(T, t1, t2, p1, p2, m1, m2) - intcspline(s, t1, t2, p1, p2, m1, m2)

struct CSpline{T}
    s; t; x::T; y::T; mx; my
end

"""
    CSpline(s, t, x, y = x, m0 = (y-x)/(t-s), m1 = (y-x)/(t-s))

Cubic spline parametrized by ``f(s) = x`` and ``f(t) = y``, ``f'(s) = m_0``, ``f'(t) = m_1``.
"""
CSpline(s, t, x::T, y = x, m0 = (y-x)/(t-s), m1 =  (y-x)/(t-s)) where {T} = CSpline{T}(s, t, x, y, mx, my)
(cs::CSpline)(t) =  cspline(t, cs.s, cs.t, cs.x, cs.y, cs.mx, cs.my)

""" 
    integrate(cs::CSpline, s, t)
    
Integrate the cubic spline from `s` to `t`.    
"""
integrate(cs::CSpline, s, t) = intcspline(s,t, cs.s, cs.t, cs.x, cs.y, cs.mx, cs.my)
