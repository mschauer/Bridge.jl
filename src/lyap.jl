# solve lyapunov equation (dH/dt) = B(t) H(t) + H(t) B(t)' - a(t), backward, conditional on Hend
function lyapunovpsdbackward_step(t, y, dt, P)
    B = Bridge.B(t - dt/2, P)
    ϕ = (I + 1/2*dt*B)\(I - 1/2*dt*B)
    ϕ *(y + 1/2*dt*a(t - dt, P))* ϕ' + 1/2*dt*Bridge.a(t, P)
end


lyapunovpsdbackward(t, P, Hend) = lyapunovpsdbackward!(samplepath(t, zero(Hend)), P, Hend)
function lyapunovpsdbackward!(H, P, Hend) # backward solve on grid tt, with final value Hend
    H.yy[end] = Hend
    t = H.tt
    for i in length(t)-1:-1:1
         H.yy[i] = lyapunovpsdbackward_step(t[i+1], H.yy[i+1], t[i+1] - t[i], P)
    end
    H
end
