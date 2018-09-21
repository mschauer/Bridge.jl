using Makie, GeometryTypes, Distributions
import Makie: convert_arguments
import Bridge: mcsvd3, visualize_uncertainty

convert_arguments(P::Type{<:Union{Lines,Scatter}}, X::SamplePath{<:AbstractVector}) = convert_arguments(P, X.yy)
convert_arguments(P::Type{<:Union{Lines,Scatter}}, X::SamplePath{<:Real}) = convert_arguments(P, X.tt, X.yy)

function band!(scene, x, ylower, yupper; nargs...)
    n = length(x)
    coordinates = [x ylower; x yupper]
    ns = 1:n-1
    ns2 = n+1:2n-1
    connectivity = [ns ns .+ 1 ns2;
                    ns2 ns2 .+ 1 ns .+ 1]
    mesh!(scene, coordinates, connectivity; shading = false, nargs...)
end

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

function visualize_uncertainty(scene, X, skip = 10, qu = 0.95; args...)
    c = sqrt(quantile(Chisq(3), qu))
    Xmean, Xrot, Xscal = X
    sphere = Sphere(Point3f0(0,0,0), 1.0f0)
    viz = visualize(
        (sphere, (Xmean[1:skip:end])),
        scale = c*(Xscal[1:skip:end]),      
        rotation = (Xrot[1:skip:end]);         
        args...
    ).children[]
    Makie.insert_scene!(scene, :mesh, viz, Dict(:show=>true, :camera=>:auto))
end

viridis(X, alpha = 0.9f0, maxviri = 200) = map(x->RGBA(Float32.(x)..., alpha), Bridge._viridis[round.(Int, range(1, stop=maxviri, length=length(X)))])
viridis(n::Integer, alpha = 0.9f0, maxviri = 200) = map(x->RGBA(Float32.(x)..., alpha), Bridge._viridis[round.(Int, range(1, stop=n>1 ? maxviri : 1, length=n))])

#set_perspective!(scene, perspective) = (push!(Makie.getscreen(scene).cameras[:perspective].view, perspective); scene)
#get_perspective(scene) = Makie.getscreen(scene).cameras[:perspective].view.value
