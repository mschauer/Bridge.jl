
using GLAbstraction, Colors, GeometryTypes, GLVisualize, Reactive

function showpath(follow = true)
        
    window = glscreen()
    timesignal = loop(1:2:div(length(tt),2))

    extra = 15

    Yxyz = collect(Point3f0, Iterators.flatten(Pt[i].Y.yy[1:end-1] for i in 1:m))
    XXxyz = collect(Point3f0, Iterators.flatten(XX[i].yy[1:end-1] for i in 1:m))
    XXmeanxyz = collect(Point3f0, Iterators.flatten(XXmean[i].yy[1:end-1] for i in 1:m))
    XXstdr = vcat([Point3f0.( XXstd[i][1:end-1] ) for i in 1:m]...)
    Xxyz = Point3f0.(Xtrue.yy)
    
    Vxyz = Point3f0.(V.yy)

    if follow
        phi = Float32(0); sca = 1.
    else
        sca = 1/15
        phi = Float32(0.2*2pi)
    end
    rotation = @SMatrix Float32[ sca*cos(phi)  -sca*sin(phi)    0.0       0.0
                                 sca*sin(phi)   sca*cos(phi)    0.0       0.0
                                 0.0            0.0             sca*1.0  -1.25
                                 0.0            0.0             0.0       1.0]
                                
#=
    rotation = map(timesignal) do i
            @SMatrix Float32[ sca*cos(phi)  -sca*sin(phi)    0.0     -XXmeanxyz[i][1]
            sca*sin(phi)   sca*cos(phi)    0.0       -XXmeanxyz[i][2]
            0.0            0.0             sca*1.0   -XXmeanxyz[i][3]
            0.0            0.0             0.0       1.0]       
    end
=#
    eyeposition = map(timesignal) do i
        Vec{3,Float32}(XXmeanxyz[i] )
    end

    # create the camera lookat and up vector
    #lookat = Signal(Vec3f0(0))
    lookat = map(timesignal) do i
        Vec{3,Float32}(XXmeanxyz[i+200] - 0XXmeanxyz[i])
    end
    upvector = Signal(Vec3f0(0,-1,0))

    # create a camera from these
    #camera = PerspectiveCamera(window.area, eyeposition, lookat, upvector)
    cam = PerspectiveCamera(
        Signal(Vec3f0(0)), # theta (rotate by x around cam xyz axis)
        Signal(Vec3f0(0)), # translation (translate by translation in the direction of the cam xyz axis)
        lookat, # lookat. We want to look at the middle of the cubes
        eyeposition, # camera position. We want to be on the same height, but further away in y
        upvector, #upvector
        window.area, # window area
        Signal(80f0), # Field of View
        Signal(2f0),  # Min distance (clip distance)
        Signal(100f0), # Max distance (clip distance)
    )
    X3d = visualize(
        Xxyz[1:skippoints:end], :lines,
        color = RGBA{Float32}(0.04, 0.15, 0.44, 0.6),
        model = rotation
    )

    Y3d = visualize(
        Yxyz[1:skippoints:end], :lines,
        color = RGBA{Float32}(0.34, 0.05, 0.14, 0.4),
        model = rotation
    )

    XX3d = visualize(
        XXxyz[1:skippoints:end], :lines,
        color = RGBA{Float32}(0.04, 0.35, 0.14, 0.4),
        model = rotation
    )
   
    circle = Sphere(Point2f0(0), 1.0f0)
    XXmean3d = visualize(
        (circle, XXmeanxyz[1:5extra*skippoints:end]),
        scale = Float32(2*sca)*XXstdr[1:5extra*skippoints:end],
        color = RGBA{Float32}(0.04, 0.35, 0.14, 0.15),
        model = rotation
    )

    XXdist = visualize(
        collect(Iterators.flatten(zip(XXmeanxyz[1:extra*skippoints:end],
         Xxyz[1:extra*skippoints:end-1]))), :linesegment,
        color = RGBA{Float32}(0.34, 0.05, 0.14, 0.40),
        model = rotation
    )
 
    circle2 = Sphere(Point2f0(0), 1f0)
    V3d = visualize(
        (circle2, Vxyz), 
        scale = fill(sca*0.1Point2f0(1,1), length(Vxyz)),
        color = RGBA{Float32}(0.7, 0.3, 0., 0.8),
        model = rotation
    )

    if follow 
        camera = cam
    else 
        camera = :perspective
    end
   _view(XXmean3d, window, camera=camera)
 #   _view(XX3d, window, camera=:perspective)
  # _view(Y3d, window, camera=camera)
    _view(X3d, window, camera=camera)
    _view(XXdist, window, camera=camera)
    _view(V3d, window, camera=camera)
    
    
    renderloop(window)
end

showpath()