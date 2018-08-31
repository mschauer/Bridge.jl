
@inline _F(t, x, P) = B(t, P)*x + β(t, P)
@inline _dHinv(t, K, P) = B(t, P)*K + K*B(t, P)' - a(t, P)
@inline _dK(t, K, P) = B(t, P)*K + K*B(t, P)' + a(t, P)
@inline _dPhi(t, Phi, P) = B(t, P)*Phi 


"""
    gpHinv!(K::SamplePath, P, KT=zero(T))

Precompute ``K = H^{-1}`` from ``(d/dt)K = BK + KB' + a`` for a guided proposal.
"""
gpHinv!(K::SamplePath{T}, P, KT=zero(T)) where {T} = _solvebackward!(R3(), _dHinv, K, KT, P)
gpH♢! = gpHinv!

"""
gpV!(K::SamplePath, P, KT=zero(T))

Precompute `V` from ``(d/dt)V = BV + β``, ``V_T = v`` for a guided proposal.
"""
gpV!(V::SamplePath{T}, P, v::T) where {T} = _solvebackward!(R3(), _F, V, v, P)



gpmu(tt, u, P) = solve(R3(), _F, tt, u, P)
gpK(tt, u, P) = solve(R3(), _dK, tt, u, P)

"""
    fundamental_matrix(tt, P) 

Compute fundamental solution.
"""
fundamental_matrix(tt, P, Phi0 = one(outertype(P))) = solve(R3(), _dPhi, tt, Phi0, P)
