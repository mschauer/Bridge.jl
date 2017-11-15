
using GLAbstraction, Colors, GeometryTypes, GLVisualize, Reactive

function showpath(;follow = true, movie = false, truth = true, obs = true, smooth = true, sample = true, ode = true, nu = true, rotating = false)
        
    window = glscreen()
    ss = 1:1:div(length(tt),2)
    timesignal = loop(ss)

    extra = 20

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

    if rotating
        rotation = map(timesignal) do i
            phi = 20tt[i]
            @SMatrix Float32[ sca*cos(phi)  -sca*sin(phi)    0.0     0.0
            sca*sin(phi)   sca*cos(phi)    0.0       0.0
            0.0            0.0             sca*1.0   -1.25
            0.0            0.0             0.0       1.0]       
        end
    else
        rotation = @SMatrix Float32[ sca*cos(phi)  -sca*sin(phi)    0.0       0.0
                                 sca*sin(phi)   sca*cos(phi)    0.0       0.0
                                 0.0            0.0             sca*1.0  -1.25
                                 0.0            0.0             0.0       1.0]
                                
    end
  
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
    if movie
        Xcolor = map(timesignal) do i; (500 < i < 900 || i > 3000)* RGBA{Float32}(0.04, 0.15, 0.44, 0.4) end
        Xsmcolor = map(timesignal) do i; (i < 400)* RGBA{Float32}(0.04, 0.15, 0.44, 0.4) end
        Ycolor = map(timesignal) do i; (1500 < i < 1900) * RGBA{Float32}(0.34, 0.05, 0.14, 0.4) end
        XXcolor =  map(timesignal) do i; (2000 < i < 2400)*RGBA{Float32}(0.04, 0.35, 0.14, 0.4) end
        XXmeancolor = map(timesignal) do i; (i > 2500)*RGBA{Float32}(0.04, 0.35, 0.14, 0.15) end
        XXdistcolor = map(timesignal) do i; (i > 3500)*RGBA{Float32}(0.34, 0.05, 0.14, 0.20) end
        Vcolor = map(timesignal) do i; (1000 < i < 3000)*RGBA{Float32}(0.7, 0.3, 0., 0.8) end

    else
        Xcolor = RGBA{Float32}(0.04, 0.15, 0.44, 0.6)
        Xsmcolor = RGBA{Float32}(0.04, 0.15, 0.44, 0.6)
        Ycolor = RGBA{Float32}(0.34, 0.05, 0.14, 0.4)
        XXcolor = RGBA{Float32}(0.04, 0.35, 0.14, 0.4)
        XXmeancolor = RGBA{Float32}(0.04, 0.35, 0.14, 0.15)
        XXdistcolor =  RGBA{Float32}(0.34, 0.05, 0.14, 0.40)
        Vcolor = RGBA{Float32}(0.7, 0.3, 0., 0.8)
    end

    X3d = visualize(
        Xxyz[1:skippoints:end], :lines,
        color = Xcolor, 
        model = rotation
    )
    Xsm3d = visualize(
        Point3f0.(X2.yy[1:skippoints:end]), :lines,
        color = Xsmcolor,
        model = rotation
    )

    Y3d = visualize(
        Yxyz[1:skippoints:end], :lines,
        color = Ycolor,
        model = rotation
    )

    XX3d = visualize(
        XXxyz[1:skippoints:end], :lines,
        color = XXcolor,
        model = rotation
    )
   
    circle = Sphere(Point2f0(0), 1.0f0)
    XXmean3d = visualize(
        (circle, XXmeanxyz[1:5extra*skippoints:end]),
        scale = Float32(2*sca)*XXstdr[1:5extra*skippoints:end],
        color = XXmeancolor,
        model = rotation
    )

    XXdist = visualize(
        collect(Iterators.flatten(zip(XXmeanxyz[1:extra*skippoints:end],
         Xxyz[1:extra*skippoints:end-1]))), :linesegment,
        color = XXdistcolor,
        model = rotation
    )
 
    circle2 = Sphere(Point2f0(0), 1f0)
    V3d = visualize(
        (circle2, Vxyz), 
        scale = fill(sca*0.5Point2f0(1,1), length(Vxyz)),
        color = Vcolor,
        model = rotation
    )

    if follow 
        camera = cam
    else 
        camera = :perspective
    end
    if smooth
        _view(XXmean3d, window, camera=camera)
    end
    if sample
        _view(XX3d, window, camera=:perspective)
    end
    if nu
        _view(Y3d, window, camera=camera)
    end

    if ode
        _view(Xsm3d, window, camera=camera)
    end
    if truth
        _view(X3d, window, camera=camera)
    end
    if truth && smooth
        _view(XXdist, window, camera=camera)
    end
    if obs 
        _view(V3d, window, camera=camera)
    end
    
    renderloop(window)
end

showpath(follow = false, rotating = true)