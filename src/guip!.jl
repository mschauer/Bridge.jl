#################################################
abstract type GuidedBridge!{T} <: ContinuousTimeProcess{T} end


function Bridge.llikelihood(::Bridge.LeftRule, Xcirc::VSamplePath, Po::Bridge.GuidedBridge!; skip = 0)
    tt = Xcirc.tt
    xx = Xcirc.yy

    som::Float64 = 0.
    x = copy(xx[.., 1]) # vector of vectors?
    tmp = copy(x)
    tmp2 = copy(x)
    r = copy(x)

    for i in 1:length(tt)-1-skip #skip last value, summing over n-1 elements
        s = tt[i]
        for k in eachindex(x)
            @inbounds x[k] = xx[k, i]
        end
        r!((i,s), x, r, Po)
        _b!((i,s), x, tmp, target(Po))
        _b!((i,s), x, tmp2, auxiliary(Po))
        tmp .-= tmp2
        som += dot(tmp, r) * (tt[i+1]-tt[i])

        if !Bridge.constdiff(Po)
            error("not implemented")
        end
    end
    som
end
