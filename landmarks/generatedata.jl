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

Example
    x0, xobs0, xobsT, Xf, P = generatedata(dataset,P,t,σobs)

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
    end
    if dataset in ["shifted","shiftedextreme"] # first stretch, then rotate, then shift; finally add noise
        q0 = [PointF(2.0cos(t), sin(t))  for t in (0:(2pi/n):2pi)[1:n]]  #q0 = circshift(q0, (1,))
        p0 = [PointF(1.0, -3.0) for i in 1:n]/n  # #p0 = [randn(Point) for i in 1:n]
        x0 = State(q0, p0)
        @time Wf, Xf = landmarksforward(t, x0, P)
        xobs0 = x0.q + σobs * randn(PointF,n)
        if dataset == "shifted" θ, η =  π/10, 0.2 end
        if dataset == "shiftedextreme" θ, η =  π/5, 0.4 end
        rot =  SMatrix{2,2}(cos(θ), sin(θ), -sin(θ), cos(θ))
        stretch = SMatrix{2,2}(1.0 + η, 0.0, 0.0, 1.0 - η)
        shift =  PointF(0.1,-0.1)
        xobsT = [rot * stretch * xobs0[i] + shift  for i in 1:P.n ] + σobs * randn(PointF,n)
    end
    if dataset=="bear"
        cd("/Users/Frank/github/BridgeLandmarks/landmarks/beardata")
        bear0 = readdlm("bear1.csv",',')
        bearT = readdlm("bear2.csv",',')
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
    end
    if dataset=="generatedstefan"
        cd("/Users/Frank/.julia/dev/Bridge/landmarks/data-stefan")

        testshapes = npzread("match.npy.npz")
        xobs0vec =  get(testshapes,"q0",0)
        xobsTvec =  get(testshapes,"v",0)
        p0vec = get(testshapes,"p",0)
        nb = div(length(xobs0vec),2)

        subs = 1:4:nb#1:5:nb
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
        plotlandmarkpositions(Xf,P,xobs0,xobsT;db=4)

    end
    x0, xobs0, xobsT, Xf, P
end

if false
    cd("/Users/Frank/.julia/dev/Bridge/landmarks/data-stefan")
    cc = npzread("cc.npy") # corpus callosum data
    cardiac = npzread("cardiac.npy")  # heart data (left ventricles, the one we used in https://arxiv.org/abs/1705.10943

end
