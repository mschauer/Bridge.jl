
function bridgeold!(Y::SamplePath, W::SamplePath, P, scheme!)
    !(W.tt[1] == P.t0 && W.tt[end] == P.t1) && error("Time axis mismatch between bridge P and driving W.") # not strictly an error
    scheme!(Y, P.v0, W, P)
    Y.yy[.., length(W.tt)] = P.v1
    Y
end

function bridgeold!(Y::SamplePath, W::SamplePath, P)
    !(W.tt[1] == P.t0 && W.tt[end] == P.t1) && error("Time axis mismatch between bridge P and driving W.") # not strictly an error
    solve!(Euler(), Y, P.v0, W, P)
    Y.yy[.., length(W.tt)] = P.v1
    Y
end

@deprecate bridge(W::SamplePath, P) solve(Euler(), W, P)
@deprecate bridge!(Y::SamplePath, W::SamplePath, P) solve!(Euler(), Y, W, P)
