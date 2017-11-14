
using GLAbstraction, Colors, GeometryTypes, GLVisualize, Reactive

function showpath()
        
    window = glscreen()
    timesignal = bounce(linspace(0.0, 1.0, 360))



    shi = ℝ{3}(0,0,1)
    sca = 1/15

    Y = vcat([Pt[i].Y[1:end-1] for i in 1:m]...)
    Yxyz = Point3f0[]
    Yintensities = Float32[]
    append!(Yxyz, map(Point3f0, Y.yy*sca .- Scalar(shi)))
    append!(Yintensities, fill(10, length(Y.tt)))

    Xxyz = Point3f0[]
    Xintensities = Float32[]
    append!(Xxyz, map(Point3f0,Xtrue.yy*sca .- Scalar(shi)))
    append!(Xintensities, fill(10, length(Xtrue.tt)))


    XXall = vcat([XX[i][1:end-1] for i in 1:m]...)
    XXxyz = Point3f0[]
    XXintensities = Float32[]
    append!(XXxyz, map(Point3f0, XXall.yy*sca .- Scalar(shi)))
    append!(XXintensities, fill(10, length(XXall.tt)))

    XXmeanall = vcat([XXmean[i][1:end-1] for i in 1:m]...)
    XXmeanxyz = Point3f0[]
    append!(XXmeanxyz, map(Point3f0, XXmeanall.yy*sca .- Scalar(shi)))
    XXstdr = vcat([Point3f0.( XXstd[i][1:end-1]*sca ) for i in 1:m]...)

    Vxyz = Point3f0[]
    Vintensities = Float32[]
    append!(Vxyz, map(Point3f0,V.yy*sca .- Scalar(shi)))
    append!(Vintensities, fill(10, length(V.tt)))


    # map comes from Reactive.jl and allows you to map any Signal to another.
    # In this case we create a rotation matrix from the timesignal signal.

    rotation = map(timesignal) do t
        rotationmatrix_z(Float32(t*2pi)) # -> 4x4 Float32 rotation matrix
    end
    rotation = rotationmatrix_z(Float32(0.2*2pi)) 
    # creates a color map from which we can sample for each line
    # and add some transparency
    if npath ==1
        cmap = [RGBA{Float64}(0.04, 0.15,0.44, 0.4)]
    else
        cmap = map(x-> RGBA{Float32}(x, 0.4), colormap("Blues", npath))
    end

    X3d = visualize(
        Xxyz[1:skippoints:end], :lines,
        intensity = Xintensities[1:skippoints:end],
        color_map = cmap,
        color_norm = Vec2f0(0, npath), # normalize intensities. Lookup in cmap will be between 0-1
        model = rotation
    )

    Y3d = visualize(
        Yxyz[1:skippoints:end], :lines,
        intensity = Yintensities[1:skippoints:end],
        color_map = [RGBA{Float64}(0.34, 0.05,0.14, 0.4)],
        color_norm = Vec2f0(0, 1), # normalize intensities. Lookup in cmap will be between 0-1
        model = rotation
    )

    XX3d = visualize(
        XXxyz[1:skippoints:end], :lines,
        intensity = XXintensities[1:skippoints:end],
        color_map = [RGBA{Float64}(0.04, 0.35, 0.14, 0.4)],
        color_norm = Vec2f0(0, 1), # normalize intensities. Lookup in cmap will be between 0-1
        model = rotation
    )
   
    circle = Sphere(Point2f0(0), 1.0f0)
    XXmean3d = visualize(
        (circle, XXmeanxyz[1:10skippoints:end]),
        scale = 2XXstdr[1:10skippoints:end],
        color_map = [RGBA{Float64}(0.04, 0.35, 0.14, 0.15)],
        color_norm = Vec2f0(0, 1), # normalize intensities. Lookup in cmap will be between 0-1
        model = rotation
    )
 

    circle2 = Sphere(Point2f0(0), 0.02f0)
    V3d = visualize(
        (circle2, Vxyz), 
        scale = fill(0.02Point2f0(1,1), length(Vxyz)),
        intensity = Vintensities,
        color_map = [RGBA{Float32}(0.2,0.0,0.6, 0.8)],
        color_norm = Vec2f0(0, 1), # normalize intensities. Lookup in cmap will be between 0-1
        model = rotation
    )

    _view(XXmean3d, window, camera=:perspective)
    _view(XX3d, window, camera=:perspective)
    _view(Y3d, window, camera=:perspective)
    _view(X3d, window, camera=:perspective)
    _view(V3d, window, camera=:perspective)
    
    
    renderloop(window)
end

showpath()