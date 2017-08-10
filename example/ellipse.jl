#https://scicomp.stackexchange.com/questions/8899/robust-algorithm-for-2x2-svd
function ellipse(x0, a, s = 1.0)
    U, S, V = svd(a)
    n = 100
    x = zeros(2, n)
    t = linspace(0, 2pi, n)
    
    u = U[:,1]
    v = U[:,2]

    for i in 1:n
                x[:,i] = x0 + s*S[1]*cos(t[i])*u + s*S[2]*sin(t[i])*v
    end

    plot(x[1,:], x[2,:])
    
end

function ellipses(mu, k, s = 0.05)
    for i in 1:length(mu)
        ellipse(mu[i], k[i], s)
    end
end


