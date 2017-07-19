using StaticArrays

function sumlogdiag{m,T}(A::SMatrix{m,m,T}, d=m) 
    t = zero(T)
    for i in 1:m
        t += log(A[i,i])
    end
    t
end   


 
