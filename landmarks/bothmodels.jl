function hamiltonian((q, p), P)
    s = 0.0
    for i in eachindex(q), j in eachindex(q)
        s += 1/2*dot(p[i], p[j])*kernel(q[i] - q[j], P)
    end
    s
end

"""
kernel in Hamiltonian
"""
kernel(q, P::Union{Landmarks, LandmarksAux, MarslandShardlow, MarslandShardlowAux}) =
 (2*π*P.a^2)^(-d/2)*exp(-norm(q)^2/(2*P.a^2))
∇kernel(q, P::Union{Landmarks, LandmarksAux, MarslandShardlow, MarslandShardlowAux}) = -P.a^(-2) * kernel(q,P) * q

"""
Needed for b! in case P is auxiliary process
"""
∇kernel(q,qT, P::Union{Landmarks, LandmarksAux, MarslandShardlow, MarslandShardlowAux}) = -P.a^(-2) * kernel(qT,P) * q

Bridge.b(t::Float64, x, P::Union{Landmarks,MarslandShardlow})= Bridge.b!(t, x, copy(x), P)

Bridge.b(t::Float64, x, P::Union{LandmarksAux,MarslandShardlowAux})= Bridge.b!(t, x, copy(x), P)


"""
Evaluate drift of landmarks in (t,x) and save to out
x is a state and out as well
"""
function Bridge.b!(t, x, out, P::MarslandShardlow)
    zero!(out)
    for i in 1:P.n
        for j in 1:P.n
            out.q[i] += p(x,j)*kernel(q(x,i) - q(x,j), P)
            # heat bath
            out.p[i] += -P.λ*p(x,j)*kernel(q(x,i) - q(x,j), P) -
                 dot(p(x,i), p(x,j)) * ∇kernel(q(x,i) - q(x,j), P)
        end
    end
    out
end

"""
Evaluate drift of landmarks in (t,x) and save to out
x is a state and out as well
"""
function Bridge.b!(t, x, out, P::Landmarks)
    zero!(out)
    for i in 1:P.n
        for j in 1:P.n
            out.q[i] += p(x,j)*kernel(q(x,i) - q(x,j), P)
            out.p[i] +=  -dot(p(x,i), p(x,j)) * ∇kernel(q(x,i) - q(x,j), P)
        end
        for k in 1:length(P.nfs)
            nf = P.nfs[k]
            out.q[i] -= 0.5 * nf.τ^(-2) * dot(q(x,i),nf.λ)  * K̄(q(x,i),nf.τ)^2 * nf.λ
            out.p[i] += 0.5 * nf.τ^(-2) * dot(p(x,i),nf.λ)  * K̄(q(x,i),nf.τ)^2 * nf.λ
        end
    end
    out
end


"""
Evaluate drift of landmarks auxiliary process in (t,x) and save to out
x is a state and out as well
"""
function Bridge.b!(t, x, out, Paux::MarslandShardlowAux)
    zero!(out)
    for i in 1:Paux.n
        for j in 1:Paux.n
            out.q[i] += p(x,j)*kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)
            # out.p[i] +=   -P.λ*0.5*p(x,j)*kernel(q(P.xT,i) - q(P.xT,j), P)
            #     -0.5* dot(p(P.xT,i), p(P.xT,j)) * ∇kernel(q(x,i) - q(x,j),q(P.xT,i) - q(P.xT,j), P)
        end
    end
    out
end

"""
Evaluate drift of landmarks auxiliary process in (t,x) and save to out
x is a state and out as well
"""
function Bridge.b!(t, x, out, Paux::LandmarksAux)
    zero!(out)
    for i in 1:Paux.n
        for j in 1:Paux.n
            out.q[i] += p(x,j)*kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)
        end
            # out.p[i] +=   -P.λ*0.5*p(x,j)*kernel(q(P.xT,i) - q(P.xT,j), P)
            #     -0.5* dot(p(P.xT,i), p(P.xT,j)) * ∇kernel(q(x,i) - q(x,j),q(P.xT,i) - q(P.xT,j), P)
        for k in 1:length(Paux.nfs)
                nf = P.nfs[k]
                out.q[i] -= 0.5 * nf.τ^(-2) * dot(q(x,i),nf.λ)  * K̄(q(Paux.xT,i),nf.τ)^2 * nf.λ
                out.p[i] += 0.5 * nf.τ^(-2) * dot(p(x,i),nf.λ)  * K̄(q(Paux.xT,i),nf.τ)^2 * nf.λ
        end
    end
    out
end


"""
Compute tildeB(t) for landmarks auxiliary process
"""
function Bridge.B(t, Paux::MarslandShardlowAux)
    I = Int[]
    J = Int[]
    X = Unc[]
    for i in 1:Paux.n
        for j in 1:Paux.n
            # terms for out.q[i]
            push!(I, 2i - 1)
            push!(J, 2j)
            push!(X, kernel(q(Paux.xT,i) - q(Paux.xT,j), P)*one(Unc))

            # terms for out.p[i]
            # push!(I, 2i)
            # push!(J, 2j-1)
            # if j==i
            #     push!(X, sum([1/(2*Paux.a) * dot(p(Paux.xT,i), p(Paux.xT,j)) * kernel(q(Paux.xT,i) - q(Paux.xT,j), P)  for j in setdiff(1:Paux.n,i)]) * one(Unc))
            # else
            #     push!(X, -1/(2*Paux.a) * dot(p(Paux.xT,i), p(Paux.xT,j)) * kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)*one(Unc))
            # end
        end
    end
    B = sparse(I, J, X, 2Paux.n, 2Paux.n)
end

"""
Compute tildeB(t) for landmarks auxiliary process
"""
function Bridge.B(t, Paux::LandmarksAux)
    I = Int[]
    J = Int[]
    X = Unc[]
    for i in 1:Paux.n
        for j in 1:Paux.n
            # terms for out.q[i]
            push!(I, 2i - 1)
            push!(J, 2j)
            push!(X, kernel(q(Paux.xT,i) - q(Paux.xT,j), P)*one(Unc))

            global out1 = zero(Unc)
            #out2 = zero(Unc)
            for k in 1:length(Paux.nfs)
                nf = P.nfs[k]
                out1 -= 0.5 * nf.τ^(-2) * K̄(q(Paux.xT,i)-nf.δ,nf.τ)^2 * nf.λ * nf.λ'
                #out2 += 0.5 * nf.τ^(-2) * K̄(q(Paux.xT,i),nf.τ)^2 * nf.λ * nf.λ'
            end
            push!(I, 2i - 1)
            push!(J, 2j - 1)
            push!(X, out1)
            push!(I, 2i)
            push!(J, 2j)
            push!(X, -out1)

            # terms for out.p[i]
            # push!(I, 2i)
            # push!(J, 2j-1)
            # if j==i
            #     push!(X, sum([1/(2*Paux.a) * dot(p(Paux.xT,i), p(Paux.xT,j)) * kernel(q(Paux.xT,i) - q(Paux.xT,j), P)  for j in setdiff(1:Paux.n,i)]) * one(Unc))
            # else
            #     push!(X, -1/(2*Paux.a) * dot(p(Paux.xT,i), p(Paux.xT,j)) * kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)*one(Unc))
            # end
        end
    end
    B = sparse(I, J, X, 2Paux.n, 2Paux.n)
end


"""
Compute B̃(t) * X (B̃ from auxiliary process) and write to out
Both B̃(t) and X are of type UncMat
"""
function Bridge.B!(t,X,out, Paux::MarslandShardlowAux)
    out .= 0.0 * out
    for i in 1:Paux.n  # separately loop over even and odd indices
        for k in 1:2Paux.n # loop over all columns
            for j in 1:Paux.n
                 out[2i-1,k] += kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)*X[p(j), k] #*one(Unc)
                # if j==i
                #    out[2i,k] +=  X[q(j),k] *sum([1/(2*Paux.a) * dot(p(Paux.xT,i), p(Paux.xT,j)) *
                #                         kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)  for j in setdiff(1:Paux.n,i)])*one(Unc)
                # else
                #    out[2i,k] +=  X[q(j),k] *(-1/(2*Paux.a)) *  dot(p(Paux.xT,i), p(Paux.xT,j)) * kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)*one(Unc)
                # end
            end
        end
    end

    out
end

"""
Compute B̃(t) * X (B̃ from auxiliary process) and write to out
Both B̃(t) and X are of type UncMat
"""
function Bridge.B!(t,X,out, Paux::LandmarksAux)

    out .= 0.0 * out
    quickway = true
    if quickway
        out = Matrix(Bridge.B(t,Paux)) * X
    else
        for i in 1:Paux.n  # separately loop over even and odd indices
            for k in 1:2Paux.n # loop over all columns
                for j in 1:Paux.n
                     out[2i-1,k] += kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)*X[p(j), k] #*one(Unc)
                    # if j==i
                    #    out[2i,k] +=  X[q(j),k] *sum([1/(2*Paux.a) * dot(p(Paux.xT,i), p(Paux.xT,j)) *
                    #                         kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)  for j in setdiff(1:Paux.n,i)])*one(Unc)
                    # else
                    #    out[2i,k] +=  X[q(j),k] *(-1/(2*Paux.a)) *  dot(p(Paux.xT,i), p(Paux.xT,j)) * kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)*one(Unc)
                    # end
                end
            end
        end
    end
    out
end


if TEST
    Hend⁺ = [rand(Unc) for i in 1:2Paux.n, j in 1:2Paux.n]
    t0 = 2.0
    BB = Matrix(Bridge.B(t0,Paux)) * Hend⁺
    out = deepcopy(Hend⁺)
    Bridge.B!(t0,Hend⁺,out,Paux)
    @test out==BB
end
