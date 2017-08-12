using StaticArrays

function sumlogdiag(A::SMatrix{m,m,T}, d=m) where {m,T} 
    t = zero(T)
    for i in 1:m
        t += log(A[i,i])
    end
    t
end   


 
