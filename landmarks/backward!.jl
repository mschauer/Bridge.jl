function Bridge.gpupdate(ν::Vector, P::Matrix, Σ, L, v)
    if all(diag(P) .== Inf)
        P_ = inv(L' * inv(Σ) * L)
        V_ = (L' * inv(Σ) * L)\(L' * inv(Σ) *  v)
        return V_, P_
    else
        Z = I - P*L'*inv(Σ + L*P*L')*L
        return Z*P*L'*inv(Σ)*v + Z*ν, Z*P
    end
end

# ms model: start with outcommented stuff in msmodel
# update/adjust partial bridgenuH file for solving backward
# update/adjust partialbridge_bolus3 to do par estimation and bridge imputation
