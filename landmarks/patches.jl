import LinearAlgebra: conj!


"""
    Initialise SamplePath on time grid t by copying x into each value of the field yy
"""
#initSamplePath(t,x) = Bridge.samplepath(t,  x)
initSamplePath(t,x) = SamplePath(t, [copy(x) for s in t])

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


function Bridge.solve(solver::EulerMaruyama!, u, W::SamplePath, P::Bridge.ProcessOrCoefficients)
    N = length(W)

    tt = W.tt
    yy = [copy(u)]

    i = 1
    dw = W.yy[i+1] - W.yy[i]
    t¯ = tt[i]
    dt = tt[i+1] - t¯

    tmp1 = Bridge._b!((i,t¯), u, 0*u, P)
    tmp2 = Bridge.σ(t¯, u, dw, P)

    y = u + tmp1*dt + tmp2

    #dump(y)
    #error("here")


    for i in 2:N-1
        t¯ = tt[i]
        dt = tt[i+1] - t¯
        push!(yy, y)
        if dw isa Number
            dw = W.yy[i+1] - W.yy[i]
        else
            for k in eachindex(dw)
                dw[k] = W.yy[i+1][k] - W.yy[i][k]
            end
        end

        Bridge._b!((i,t¯), y, tmp1, P)
        Bridge.σ!(t¯, y, dw, tmp2, P)

        for k in eachindex(y)
            y[k] = y[k] + tmp1[k]*dt + tmp2[k]
        end
    end
    copyto!(yy[end], Bridge.endpoint(y, P))
    SamplePath(tt, yy)
end


function Bridge.solve!(solver::EulerMaruyama!, Y::SamplePath, u, W::SamplePath, P::Bridge.ProcessOrCoefficients)
    N = length(W)

    tt = W.tt
    yy = Y.yy
    copyto!(yy[1], u)

    i = 1
    dw = W.yy[i+1] - W.yy[i]
    t¯ = tt[i]
    dt = tt[i+1] - t¯

    tmp1 = Bridge._b!((i,t¯), u, 0*u, P)
    tmp2 = Bridge.σ(t¯, u, dw, P)

    y = u + tmp1*dt + tmp2

    #dump(y)
    #error("here")


    for i in 2:N-1
        t¯ = tt[i]
        dt = tt[i+1] - t¯
        copyto!(yy[i], y)
        if dw isa Number
            dw = W.yy[i+1] - W.yy[i]
        else
            for k in eachindex(dw)
                dw[k] = W.yy[i+1][k] - W.yy[i][k]
            end
        end

        Bridge._b!((i,t¯), y, tmp1, P)
        Bridge.σ!(t¯, y, dw, tmp2, P)

        for k in eachindex(y)
            y[k] = y[k] + tmp1[k]*dt + tmp2[k]
        end
    end
    copyto!(yy[end], Bridge.endpoint(y, P))
    SamplePath(tt, yy)
end


struct StratonovichHeun! <: Bridge.SDESolver
end

function Bridge.solve!(solver::StratonovichHeun!, Y::SamplePath, u, W::SamplePath, P::Bridge.ProcessOrCoefficients)
    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    tt = Y.tt
    tt[:] = W.tt
    yy = Y.yy
    y = copy(u)
    ȳ = copy(u)

    tmp1 = copy(y)
    tmp2 = copy(y)
    tmp3 = copy(y)
    tmp4 = copy(y)

    dw = copy(W.yy[1])
    for i in 1:N-1
        t¯ = tt[i]
        dt = tt[i+1] - t¯
        copyto!(yy[i], y)
        if dw isa Number
            dw = W.yy[i+1] - W.yy[i]
        else
            for k in eachindex(dw)
                dw[k] = W.yy[i+1][k] - W.yy[i][k]
            end
        end

        Bridge._b!((i,t¯), y, tmp1, P)
        Bridge.σ!(t¯, y, dw, tmp2, P)

        for k in eachindex(y)
            ȳ[k] = y[k] + tmp1[k]*dt + tmp2[k] # Euler prediction
        end

        Bridge._b!((i + 1,t¯ + dt), ȳ, tmp3, P) # coefficients at ȳ
        #Bridge.σ!(t¯ + dt, ȳ, dw2, tmp4, P)  # original implementation
        Bridge.σ!(t¯ + dt, ȳ, dw, tmp4, P)

        for k in eachindex(y)
            y[k] = y[k] + 0.5*((tmp1[k] + tmp3[k])*dt + tmp2[k] + tmp4[k])
        end
    end
    copyto!(yy[end], Bridge.endpoint(y, P))
    Y
end

function LinearAlgebra.naivesub!(At::Adjoint{<:Any,<:LowerTriangular}, b::AbstractVector, x::AbstractVector = b)
    A = At.parent
    n = size(A, 2)
    if !(n == length(b) == length(x))
        throw(DimensionMismatch("second dimension of left hand side A, $n, length of output x, $(length(x)), and length of right hand side b, $(length(b)), must be equal"))
    end
    @inbounds for j in n:-1:1
        iszero(A.data[j,j]) && throw(SingularException(j))
        xj = x[j] = A.data[j,j] \ b[j]
        for i in j-1:-1:1 # counterintuitively 1:j-1 performs slightly better
            b[i] -= A.data[j,i] * xj
        end
    end
    x
end

function LinearAlgebra.naivesub!(At::Adjoint{<:Any,<:LowerTriangular}, B::AbstractMatrix, X::AbstractMatrix = B)
    A = At.parent
    n = size(A, 2)
    if !(n == size(B,1) == size(B,2) == size(X,1) == size(X,2))
        throw(DimensionMismatch())
    end
    @inbounds for k in 1:n
        for j in n:-1:1
            iszero(A.data[j,j]) && throw(SingularException(j))
            xjk = X[j,k] = A.data[j,j] \ B[j,k]
            for i in j-1:-1:1 # counterintuitively 1:j-1 performs slightly better
                B[i,k] -= A.data[j,i] * xjk
            end
        end
    end
    X
end

function LinearAlgebra.naivesub!(A::LowerTriangular, B::AbstractMatrix, X::AbstractMatrix = B)
    n = size(A,2)
    if !(n == size(B,1) == size(X,1))
        throw(DimensionMismatch())
    end
    if !(size(B,2) == size(X,2))
        throw(DimensionMismatch())
    end


    @inbounds for k in 1:size(B,2)
        for j in 1:n
            iszero(A.data[j,j]) && throw(SingularException(j))
            xjk = X[j,k] = A.data[j,j] \ B[j,k]
            for i in j+1:n
                B[i,k] -= A.data[i,j] * xjk
            end
        end
    end
    X
end


function LinearAlgebra.naivesub!(A::UpperTriangular, B::AbstractMatrix, X::AbstractMatrix = B)
    n = size(A, 2)
    if !(n == size(B,1) == size(X,1))
        throw(DimensionMismatch())
    end
    if !(size(B,2) == size(X,2))
        throw(DimensionMismatch())
    end

    @inbounds for k in 1:size(B, 2)
        for j in n:-1:1
            iszero(A.data[j,j]) && throw(SingularException(j))
            xjk = X[j,k] = A.data[j,j] \ B[j,k]
            for i in j-1:-1:1 # counterintuitively 1:j-1 performs slightly better
                B[i,k] -= A.data[i,j] * xjk
            end
        end
    end
    X
end

function lyapunovpsdbackward_step!(t, dt, Paux,Hend⁺,H⁺)
    B = Matrix(Bridge.B(t - dt/2, Paux))
    ϕ = (I + 1/2*dt*B)\(I - 1/2*dt*B)
    #Ht .= ϕ *(Hend⁺ + 1/2*dt*Bridge.a(t - dt, Paux))* ϕ' + 1/2*dt*Bridge.a(t, Paux)
    H⁺ .= ϕ *(Hend⁺ + 1/2*dt*Bridge.a(t - dt, Paux))* conj!(copy(ϕ)) + 1/2*dt*Bridge.a(t, Paux)
    H⁺
end

"""
Version where B̃ and ã do not depend on try
"""
function lyapunovpsdbackward_step!(t, dt, Paux,Hend⁺,H⁺,B̃, ã)
    ϕ = (I + 1/2*dt*B̃)\(I - 1/2*dt*B̃)
    #Ht .= ϕ *(Hend⁺ + 1/2*dt*Bridge.a(t - dt, Paux))* ϕ' + 1/2*dt*Bridge.a(t, Paux)
    H⁺ .= ϕ *(Hend⁺ + 1/2*dt*ã)* conj!(copy(ϕ)) + 1/2*dt*ã
    H⁺
end


"""
Compute transpose of square matrix of Unc matrices

A = reshape([Unc(1:4), Unc(5:8), Unc(9:12), Unc(13:16)],2,2)
B = copy(A)
A
conj!(B)
"""
function conj!(A::Array{<:Unc,2}) ## FIX, no copy of A wanted
    m, n = size(A)
    B = reshape(copy(A), n, m)
    for i in 1:m
        for j in 1:n
                B[j, i] = A[i, j]'
        end
    end
    B
end




function conj2(A::Array{T,2}) where {T<:Unc}
    At =  Matrix{T}(undef,size(A,2),size(A,1))
    for i in 1:size(A,2)
        for j in 1:size(A,1)
            At[i,j] = A[j,i]'
        end
    end
    At
end

if TEST
        A = reshape(rand(Unc,15),5,3)
        B = conj2(A)
        @test norm(deepmat(A)'-deepmat(B))<10^(-10)
end


"""
    Forward simulate landmarks process specified by P on grid t.
    Returns driving motion W and landmarks process X
    t: time grid
    x0: starting point
    P: landmarks specification
"""
function landmarksforward(t, x0::State{Pnt}, P) where Pnt
    W = initSamplePath(t,  zeros(Pnt, dimwiener(P)))
    sample!(W, Wiener{Vector{Pnt}}())
    # forward simulate landmarks
    X = initSamplePath(t,x0)
    #println("Solve for forward process:")
    solve!(EulerMaruyama!(), X, x0, W, P)  #@time solve!(StratonovichHeun!(), X, x0, W, P)
    W, X
end

# """
#     Forward simulate landmarks process specified by P using Wiener process W.
#     Writes into X
#     x0: starting point
#     P: landmarks specification
#     W: driving Wiener process
#     X: samplepath written into
#
#     Need that W.tt == X.tt
# """
# function landmarksforward!(x0::State{Pnt}, P, W, X) where Pnt
#     N = length(W)
#     N != length(X) && error("X and W differ in length.")
#
#     solve!(EulerMaruyama!(), X, x0, W, P)
# end


tc(t,T) = t.* (2 .-t/T)
extractcomp(v,i) = map(x->x[i], v)

"""
    Adapting Radford Neal's R implementation of Hamiltonian Monte Carlo with
    stepsize ϵ and L steps
"""
function HMC(U, ∇U, ϵ, L, current_q)
    q = current_q
    p = randn(length(q)) # independent standard normal variates
    current_p = p
    # Make a half step for momentum at the beginning
    p = p - ϵ * ∇U(q) / 2
    # Alternate full steps for position and momentum
    for i in 1:L
        # Make a full step for the position
        q = q + ϵ * p
        # Make a full step for the momentum, except at end of trajectory
        if !(i==L)
            p = p - ϵ * ∇U(q)
        end
    end
    # Make a half step for momentum at the end.
    p = p - ϵ * ∇U(q) / 2
    # Negate momentum at end of trajectory to make the proposal symmetric
    p = -p
    # Evaluate potential and kinetic energies at start and end of trajectory
    current_U = U(current_q)
    current_K = sum(current_pˆ2) / 2
    proposed_U = U(q)
    proposed_K = sum(pˆ2) / 2
    # Accept or reject the state at end of trajectory, returning either
    # the position at the end of the trajectory or the initial position
    if rand() < exp(current_U-proposed_U+current_K-proposed_K)
        return q # accept
    else
        return current_q # reject
    end
end


function deepcopyto!(dest::AbstractArray{T1,N}, src::AbstractArray{T2,N}) where {T1,T2,N}
    checkbounds(dest, axes(src)...)
    src′ = Base.unalias(dest, src)
    for I in eachindex(IndexStyle(src′,dest), src′)
        @inbounds dest[I] = deepcopy(src′[I])
    end
    dest
end
