using Makie, GeometryTypes
import Makie.to_positions
import Bridge: mcsvd3, visualize_uncertainty

to_positions(S::Makie.Scene, X::Bridge.SamplePath) = to_positions(S, X.yy)


"""
    mcsvd3(mcstates) -> mean, q, sv

Compute mean and decompose variance into singular values `sv` and 
3d rotation quaternion `q`. 

Returns vectors of appropriate `GeometryTypes`.
"""
function mcsvd3(states)
    Xmean = Point3f0[]
    Xscal = Point3f0[]
    Xrot = Vec4f0[]

    for i in 1:length(states)
        xx, vv = Bridge.mcstats(states[i])
        svds = svd.(vv)
        append!(Xmean, xx)
        append!(Xscal, map(svd->sqrt.(svd[2]), svds))
        append!(Xrot, Bridge.quaternion.(first.(svds)))
        if i != length(states)
            pop!(Xmean); pop!(Xscal); pop!(Xrot)
        end
    end 
    Xmean, Xrot, Xscal
end

function visualize_uncertainty(scene, X, skip = 10; args...)
    Xmean, Xrot, Xscal = X
    sphere = Sphere(Point3f0(0,0,0), 1.0f0)
    viz = visualize(
        (sphere, (Xmean[1:skip:end])),
        scale = (Xscal[1:skip:end]),      
        rotation = (Xrot[1:skip:end]);         
        args...
    ).children[]
    Makie.insert_scene!(scene, :mesh, viz, Dict(:show=>true, :camera=>:auto))
end