
using GLAbstraction, Colors, GeometryTypes, GLVisualize, Reactive

function showpath(;follow = true, movie = false, x0 = false, truth = true, obs = true, smooth = true, sample = true, ode = true, nu = true, rotating = false)
        
    window = glscreen()
    if follow
        ss = 1:2:div(length(tt),2)
    else
        ss = 1:4:5000
    end
    timesignal = loop(ss)

    extra = 10

    Yxyz = collect(Point3f0, Iterators.flatten(Pt[i].Y.yy[1:end-1] for i in 1:m)) # Y
    #Yxyz = collect(Point3f0, Iterators.flatten(Pᵒ[i].V[1:end-1] for i in 1:m)) # V

    XXxyz = collect(Point3f0, Iterators.flatten(XX[i].yy[1:end-1] for i in 1:m))
    XXmeanxyz = collect(Point3f0, Iterators.flatten(XXmean[i].yy[1:end-1] for i in 1:m))
    XXstdr = vcat([Point3f0.( XXstd[i][1:end-1] ) for i in 1:m]...)
    XXscale = vcat([Point3f0.(XXscal[i][1:end-1]) for i in 1:m]...)
    XXrotation = vcat([Vec4f0.(XXrot[i][1:end-1]) for i in 1:m]...)
    
    
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
            phi = 25.5tt[i]
            sca_ = sca+ sca*(i>3000)*(2*atan((i-3000)/1000)/pi)
            @SMatrix Float32[ sca_*cos(phi)  -sca_*sin(phi)    0.0     0.0
            sca_*sin(phi)   sca_*cos(phi)    0.0       0.0
            0.0            0.0             sca_*1.0   -sca_/sca*1.25
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
        Xvisible = map(timesignal) do i; (500 < i < 950 || i > 3000) end
        Xsmvisible = map(timesignal) do i; (i < 450) end
        Yvisible = map(timesignal) do i; (1500 < i < 1950) end
        XXvisible =  map(timesignal) do i; (2000 < i < 2450) end
        XXmeanvisible = map(timesignal) do i; (i > 2500) end
        XXdistvisible = map(timesignal) do i; (i > 3500) end
        Vvisible = map(timesignal) do i; (1000 < i < 3000) end

    else
        Xvisible = 
        Xvisible = 
        Xsmvisible = 
        Yvisible = 
        XXvisible = 
        XXmeanvisible =
        XXdistvisible =
        Vvisible = map(timesignal) do i; true end
    end
    X0visible = map(timesignal) do i; true end

    Xcolor = RGBA{Float32}(0.04, 0.15, 0.44, 0.6)
    Xsmcolor = RGBA{Float32}(0.04, 0.15, 0.44, 0.8)
    Ycolor = RGBA{Float32}(0.34, 0.05, 0.14, 0.4)
    XXcolor = RGBA{Float32}(0.04, 0.35, 0.14, 0.4)
    XXmeancolor = RGBA{Float32}(0.04, 0.35, 0.14, 0.15)
    XXdistcolor =  RGBA{Float32}(0.34, 0.05, 0.14, 0.40)
    Vcolor = RGBA{Float32}(0.7, 0.3, 0., 0.5)


    X3d = visualize(
        Xxyz[1:skippoints:end], :lines,
        color = Xcolor, 
        model = rotation,
        visible = Xvisible
    )
    Xsm3d = visualize(
        Point3f0.(X2.yy[1:skippoints:end]), :lines,
        color = Xsmcolor,
        model = rotation,
        visible = Xsmvisible
    )

    Y3d = visualize(
        Yxyz[1:skippoints:end], :lines,
        color = Ycolor,
        model = rotation,
        visible = Yvisible
    )

    XX3d = visualize(
        XXxyz[1:skippoints:end], :lines,
        color = XXcolor,
        model = rotation,
        visible = XXvisible
    )
   
    sphere = Sphere(Point3f0(0,0,0), 1.0f0)
    XXmean3d = visualize(
        (sphere, XXmeanxyz[1:5extra*skippoints:end]),
        #scale = Float32(2*sca)*XXstdr[1:5extra*skippoints:end],
        scale = Float32(1)*XXscale[1:5extra*skippoints:end],      
        rotation = XXrotation[1:5extra*skippoints:end],          
        color = XXmeancolor,
        model = rotation,
        visible = XXmeanvisible
    )

    XXdist = visualize(
        collect(Iterators.flatten(zip(XXmeanxyz[1:extra*skippoints:end],
         Xxyz[1:extra*skippoints:end-1]))), :linesegment,
        color = XXdistcolor,
        model = rotation,
        visible = XXdistvisible
    )
 
    sphere2 = Sphere(Point3f0(0), 1f0)
    V3d = visualize(
        (sphere2, Vxyz), 
        scale = fill(0.5Point3f0(1,1,1), length(Vxyz)),
        color = Vcolor,
        model = rotation,
        visible = Vvisible
    )

    sphere3 = Sphere(Point3f0(0), 0.1f0)    
    X03d = visualize(
        (sphere3, Point3f0.(X0[1:10:end])), 
        color = Xcolor,
        model = rotation,
        visible = X0visible
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
    if x0
        _view(X03d, window, camera=:perspective)
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
    
    if !movie
        renderloop(window)
    else

        # save video to report dir, or in some tmp dir we'll delete later
        path = "./output"

        isdir(path) || mkdir(path)
        name = path * "/$(randstring()).mkv"

        #@async renderloop(window)

        # create a stream to which we can add frames
        io = GLVisualize.create_video_stream(name, window)
        for i in 1:div(3900,2)
            # do something
               #render current frame
            # if you call @async renderloop(window) you can replace this part with yield
            #yield()
            GLWindow.render_frame(window)
            GLWindow.swapbuffers(window)
            GLWindow.reactive_run_till_now()

            # add the frame from the current window
            GLVisualize.add_frame!(io)
        end
        # closing the stream will trigger writing the video!
        close(io.io)
        GLWindow.destroy!(window)
    end    
end

showpath(;follow = false, movie = false, truth = true, obs = false, smooth = true, sample = false, x0 = false, ode = false, nu = true, rotating = false)

#showpath(;follow = false, movie = true, rotating = true)