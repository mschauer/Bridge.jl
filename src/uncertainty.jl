
struct EstSamplePath{T,R<:AbstractVector,S}
    tt::R
    yy::Vector{T}
    vv::Vector{S}
end

function piecewise(X::EstSamplePath, tend = X.tt[end])
    EstSamplePath(piecewise(SamplePath(X.tt, X.yy), tend)..., 
        piecewise(SamplePath(X.tt, X.vv), tend)[2])
end

struct SamplePathBand{T,R<:AbstractVector}
    tt::R
    ll::Vector{T}
    uu::Vector{T}
end

function piecewise(X::SamplePathBand, tend = X.tt[end])
    SamplePathBand(piecewise(SamplePath(X.tt, X.ll), tend)..., 
        piecewise(SamplePath(X.tt, X.uu), tend)[2])
end

