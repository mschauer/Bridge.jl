using Makie, GLVisualize
using GeometryTypes, Colors
using Bridge: _viridis

skippoints = 1
extra = 50
d2 = length(valtype(V))
scene = Scene(resolution = (400, 400))

sphere2 = Sphere(Point2f0(0), 1.0f0)
lines((@view Xtrue.yy[1:skippoints:end]), linewidth = 0.2, color = :darkblue)
#lines((@view Xtrue.yy[1:skippoints:end]), linewidth = 0.2, color = [:darkblue,:darkred][1+(map(x->dot(R{3}(1,1,0),x), Xtrue.yy[1:skippoints:end]) .> 0) ])



if d2 == 3
    viri = _viridis[round.(Int,range(1, stop=256, length=length(V)))]
    scatter(V.yy, marker=sphere2, markersize = 0.2,  strokewidth = 0.02, strokecolor = :white, color = map(x->RGBA(Float32.(x)..., 0.9f0), viri)) 
elseif d2 == 2
 #   viri = _viridis[round.(Int,linspace(1,256, 2*length(V)))]
  #  linesegment(map(v->(Point3f0(-10, v...), Point3f0(10, v...)), V.yy), color = map(x->RGBA(Float32.(x)..., 0.9f0), viri))
    viri = _viridis[round.(Int,range(1, stop=256, length=length(V)))]
    scatter(map(v->Point3f0(0, v...), V.yy), marker=sphere2, markersize = 0.2,  strokewidth = 0.02, strokecolor = :white, color = map(x->RGBA(Float32.(x)..., 0.9f0), viri)) 
end
#for pt in Pt
#    lines((@view pt.Y.yy[1:skippoints:end]), linewidth = 0.2, color = :darkred)
#end


Nu = vcat([Point3f0.(Pᵒ[i].V[1:end-1]) for i in 1:m]...)
lines((@view Nu[1:skippoints:end]), linewidth = 0.2, color = :darkred)

lines((@view Xmean[1:skippoints:end]), linewidth = 0.2, color = :darkgreen)

visualize_uncertainty(scene, (Xmean, Xrot, Xscal), extra*skippoints; color = RGBA(0.2f0, 0.4f0, 0.9f0, 0.1f0))

sphere = Sphere(Point3f0(0,0,0), 1.0f0)

if d2 == 3
    axis(map(x -> UnitRange(round.(Int,x)...), extrema(Bridge.mat(V.yy), 2))...)
else 
    #axis(map(x -> UnitRange(round.(Int,x)...), extrema(Bridge.mat(Xtrue.yy), 2))...)
    ma = maximum(Bridge.mat(X2.yy), 2) 
    axis(map(x -> UnitRange(0, round.(Int,x)), ma)..., showaxis = (true,true,true), showgrid = (false,false,false))
    
    for (x, c) in zip(ℝ{3}[(ma[1], 0, 0), (0, ma[2], 0),(0, 0, ma[3]), (0,0,0)], ['x', 'y', 'z', '•'])
        scatter([x], marker = c, markersize = 1., color = :black)
    end
end

scatter(collect(Models.foci(P)), marker=Sphere(Point2f0(0), 1.0f0), markersize = 1.,  strokewidth = 0.1, strokecolor = :white, color = :green)


center!(scene)

