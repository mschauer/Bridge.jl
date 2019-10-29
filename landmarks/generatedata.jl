"""
Generate landmarks data, or read landmarks data.
- dataset: specifies type of data to be generated/loaded
- P:
- t: grid used for generating data
- σobs: standard deviation of observation noise

Returns
- x0: initial state without noise
- xobs0, xobsT: (observed initial and final states with noise)
- Xf: forward simulated path
- P: only adjusted in case of 'real' data, like the bearskull data
- pb: plotting bounds used in animation

Example
    x0, xobs0, xobsT, Xf, P, pb = generatedata(dataset,P,t,σobs)

"""
function generatedata(dataset,P,t,σobs)

    n = P.n
    if dataset=="forwardsimulated"
        q0 = [PointF(2.0cos(t), sin(t)) for t in (0:(2pi/n):2pi)[1:n]]  #q0 = circshift(q0, (1,))
        p0 = [Point(1.0, -3.0) for i in 1:n]/n  # #p0 = [randn(Point) for i in 1:n]
        x0 = State(q0, p0)
        Wf, Xf = landmarksforward(t, x0, P)
        xobs0 = x0.q + σobs * randn(PointF,n)
        xobsT = [Xf.yy[end].q[i] for i in 1:P.n ] + σobs * randn(PointF,n)
        pb = Lmplotbounds(-2.0,2.0,-1.5,1.5)
        obs_atzero = true
    end
    if dataset in ["shifted","shiftedextreme"] # first stretch, then rotate, then shift; finally add noise
        q0 = [PointF(2.0cos(t), sin(t))  for t in (0:(2pi/n):2pi)[1:n]]  #q0 = circshift(q0, (1,))
        p0 = [PointF(1.0, -3.0) for i in 1:n]/n  # #p0 = [randn(Point) for i in 1:n]
        x0 = State(q0, p0)
        @time Wf, Xf = landmarksforward(t, x0, P)
        xobs0 = x0.q + σobs * randn(PointF,n)
        if dataset == "shifted"
            θ, η =  π/10, 0.2
            pb = Lmplotbounds(-3.0,3.0,-2.0,2.0) # verified
        end
        if dataset == "shiftedextreme"
            θ, η =  π/5, 0.4
            pb = Lmplotbounds(-3.0,3.0,-3.0,3.0)
        end
        rot =  SMatrix{2,2}(cos(θ), sin(θ), -sin(θ), cos(θ))
        stretch = SMatrix{2,2}(1.0 + η, 0.0, 0.0, 1.0 - η)
        shift =  0.0*PointF(0.1,-0.1)
        xobsT = [rot * stretch * xobs0[i] + shift  for i in 1:P.n ] + σobs * randn(PointF,n)
        obs_atzero = true
    end
    if dataset=="bear"
        bear0 = readdlm(joinpath(@__DIR__,"data-stefan", "bear1.csv"), ',')
        bearT = readdlm(joinpath(@__DIR__,"data-stefan", "bear2.csv"), ',')
        nb = size(bear0)[1]
        avePoint = Point(414.0, 290.0)  # average of (x,y)-coords for bear0 to center figure at origin
        xobs0 = [Point(bear0[i,1], bear0[i,2]) - avePoint for i in 1:nb]/200.
        xobsT = [Point(bearT[i,1], bearT[i,2]) - avePoint for i in 1:nb]/200.
        # need to redefine P, because of n
        if model == :ms
            P = MarslandShardlow(P.a, P.c, P.γ, P.λ, nb)
        else
            P = Landmarks(P.a, P.c, nb, P.db, P.nfstd, P.nfs)
        end
        x0 = State(xobs0, rand(PointF,P.n))
        Wf, Xf = landmarksforward(t, x0, P)
        pb = Lmplotbounds(-2.0,2.0,-1.5,1.5)
        obs_atzero = true
    end
    if dataset=="heart"
        q0 = [PointF(2.0cos(t), 2.0sin(t))  for t in (0:(2pi/n):2pi)[1:n]]  #q0 = circshift(q0, (1,))
        p0 = [PointF(1.0, -3.0) for i in 1:n]/n  # #p0 = [randn(Point) for i in 1:n]
        x0 = State(q0, p0)
        @time Wf, Xf = landmarksforward(t,  x0, P)
        xobs0 = x0.q + σobs * randn(PointF,n)
        heart_xcoord(s) = 0.2*(13cos(s)-5cos(2s)-2cos(3s)-cos(4s))
        heart_ycoord(s) = 0.2*16(sin(s)^3)
        qT = [PointF(heart_xcoord(t), heart_ycoord(t))  for t in (0:(2pi/n):2pi)[1:n]]  #q0 = circshift(q0, (1,))
        xobsT = qT + σobs * randn(PointF,n)
        pb = Lmplotbounds(-2.0,2.0,-1.5,1.5)
        obs_atzero = true
    end
    if dataset=="peach"
        q0 = [PointF(2.0cos(t), 2.0sin(t))  for t in (0:(2pi/n):2pi)[1:n]]  #q0 = circshift(q0, (1,))
        p0 = [PointF(1.0, -3.0) for i in 1:n]/n  # #p0 = [randn(Point) for i in 1:n]
        x0 = State(q0, p0)
        @time Wf, Xf = landmarksforward(t, x0, P)
        xobs0 = x0.q + σobs * randn(PointF,n)
        peach_xcoord(s) = (2.0 + sin(s)^3) * cos(s)
        peach_ycoord(s) = (2.0 + sin(s)^3) * sin(s)
        qT = [PointF(peach_xcoord(t), peach_ycoord(t))  for t in (0:(2pi/n):2pi)[1:n]]  #q0 = circshift(q0, (1,))
        xobsT = qT + σobs * randn(PointF,n)
        pb = Lmplotbounds(-2.0,2.0,-1.5,1.5)
        obs_atzero = true
    end
    if dataset=="generatedstefan"
        testshapes = npzread(joinpath(@__DIR__,"data-stefan", "match.npy.npz"))
        xobs0vec =  get(testshapes,"q0",0)
        xobsTvec =  get(testshapes,"v",0)
        p0vec = get(testshapes,"p",0)
        nb = div(length(xobs0vec),2)

        subs = 1:6:nb#1:5:nb
        xobs0 = [PointF(xobs0vec[2i-1],xobs0vec[2i]) for i in subs]
        xobsT = [PointF(xobsTvec[2i-1],xobsTvec[2i]) for i in subs]
        p0 = [PointF(p0vec[2i-1],p0vec[2i]) for i in subs]
        nb = length(subs)

        # need to redefine P, because of n
        if model == :ms
            P = MarslandShardlow(P.a,P.c, P.γ, P.λ, nb)
        elseif model== :ahs
            P = Landmarks(P.a,P.c, nb, P.db, P.nfstd, P.nfs)
        end
        x0 = State(xobs0, p0)
        Wf, Xf = landmarksforward(t, x0, P)
#        plotlandmarkpositions(Xf,P,xobs0,xobsT;db=4)
        pb = Lmplotbounds(-2.0,2.0,-1.5,1.5) # verified
        obs_atzero = true
    end

    if dataset=="forwardsimulated_multiple"
        nshapes = 3
        q0 = [PointF(2.0cos(t), sin(t)) for t in (0:(2pi/n):2pi)[1:n]]  #q0 = circshift(q0, (1,))
        p0 = [Point(0.1, -0.4) for i in 1:n]/n  # #p0 = [randn(Point) for i in 1:n]
        #p0 = [Point(4.1, -0.4) for i in 1:n]/n  # #p0 = [randn(Point) for i in 1:n]
        x0 = State(q0, p0)
        xobsT = Vector{PointF}[]
        for k in 1:nshapes
            #a = P.a * exp(0.15*randn())
            a = 5*P.a * exp(0.15*randn())
            c = P.c * exp(0.15*randn())
            γ = 0.6*getγ(P) * exp(0.01*randn())
            if model == :ms
                P = MarslandShardlow(a, c, γ, P.λ, P.n)
            elseif model == :ahs
                nfs = construct_nfs(P.db, P.nfstd, γ)
                P = Landmarks(a, c, P.n, P.db, P.nfstd, nfs)
            end
            Wf, Xf = landmarksforward(t, x0, P)

            push!(xobsT, [Xf.yy[end].q[i] for i in 1:P.n ] + σobs * randn(PointF,n))
        end
        xobs0 = []
        pb = Lmplotbounds(-2.0,2.0,-1.5,1.5)
        obs_atzero = false
    end
    if dataset=="cardiac"
        # see https://arxiv.org/pdf/1805.06038.pdf
        # 14 cardiac images, 66 landmarks per image
        # The epicardial and endocardial contours where annotated with 33 landmarks along each outline resulting in 66 landmarks per image
        # Each of the 14 rows is a shape with 66 landmarks, x coordinates in
        # cardiac[:,:,0], y coordinates in cardiac[:,:,1], i.e. in python, all
        # shapes are plotted with
        # for i in range(N_samples):
        #      plt.plot(qs[i,:,0],qs[i,:,1])

        cardiac = npzread(joinpath(@__DIR__,"data-stefan","cardiac.npy"))  # heart data (left ventricles, the one we used in https://arxiv.org/abs/1705.10943
        cardiacx = cardiac[:,:,1]  # x-coordinates of landmarks
        cardiacy = cardiac[:,:,2]  # y-coordinates of landmarks
        nshapes = 14
        landmarksset = 1:5:66
        n = length(landmarksset)
        xobsT = fill(zeros(PointF,66),nshapes)
        for i in 1:nshapes # for each image
            xobsT[i] = [PointF(cardiacx[i,j], cardiacy[i,j]) for j in landmarksset ]
        end
        x0 = State(zeros(PointF,n), zeros(PointF,n))
        pb = Lmplotbounds(-2.0,2.0,-1.5,1.5) # need to check
        obs_atzero = false
        xobs0 = []
         Xf = 0
        if model == :ms
            P = MarslandShardlow(P.a,P.c, P.γ, P.λ, n)
        elseif model== :ahs
            P = Landmarks(P.a,P.c, n, P.db, P.nfstd, P.nfs)
        end

        xobsTdf = DataFrame(x=extractcomp(xobsT[1],1), y =extractcomp(xobsT[1],2))
        @rput xobsTdf
        R"""
            library(tidyverse)
            library(ggplot2)
            xobsTdf %>% ggplot(aes(x=x,y=y)) + geom_point() + geom_path()
        """
        pb = Lmplotbounds(-0.25,0.25,-0.25,0.25)
    end
    x0, xobs0, xobsT, Xf, P, pb, obs_atzero
end

if false
    cc = npzread(joinpath(@__DIR__,"data-stefan","cc.npy")) # corpus callosum data

                    # there are 14 cardiac images, in https://arxiv.org/pdf/1705.10943.pdf 17 landmarks are chosen
end
