struct Lmplotbounds
    xmin::Float64
    xmax::Float64
    ymin::Float64
    ymax::Float64
end



function drawpath(i,n,x,X,objvals,parsave,(xobs0comp1,xobs0comp2,xobsTcomp1, xobsTcomp2),pb;showmomenta=false)
        s = deepvec2state(x).p
        pp1 = plotshapes(xobs0comp1,xobs0comp2,xobsTcomp1, xobsTcomp2)
        if showmomenta
            Plots.plot!(pp1, extractcomp(s,1), extractcomp(s,2),seriestype=:scatter ,
             color=:blue,label="p0 est") # points move from black to orange)
            Plots.plot!(pp1, extractcomp(s0,1), extractcomp(s0,2),seriestype=:scatter ,
              color=:red,label="p0",markersize=5) # points move from black to orange)
              xlims!(-3,3)
              ylims!(-4,3)
        else
            xlims!(pb.xmin,pb.xmax)
            ylims!(pb.ymin,pb.ymax)
        end

        outg = [Any[X.tt[i], [X.yy[i][CartesianIndex(c, k)][l] for l in 1:d, c in 1:2]..., "point$k"] for k in 1:n, i in eachindex(X.tt) ][:]
        dfg = DataFrame(time=extractcomp(outg,1),pos1=extractcomp(outg,2),pos2=extractcomp(outg,3),mom1=extractcomp(outg,4),mom2=extractcomp(outg,5),pointID=extractcomp(outg,6))
        for j in 1:n
            el1 = dfg[!,:pointID].=="point"*"$j"
            dfg1 = dfg[el1,:]
            Plots.plot!(pp1,dfg1[!,:pos1], dfg1[!,:pos2],label="")
        end

        pp2 = Plots.plot(collect(0:i), objvals[1:i+1],seriestype=:scatter ,color=:blue,markersize=1.5,label="",title="Loglikelihood approximation")
        Plots.plot!(pp2, collect(0:i), objvals[1:i+1] ,color=:blue,label="")
        xlabel!(pp2,"iteration")
        ylabel!(pp2,"stoch log likelihood")
        xlims!(0,ITER)

        avals = extractcomp(parsave,1)
        pp3 = Plots.plot(collect(0:i), avals[1:i+1],seriestype=:scatter ,color=:blue,markersize=1.5,label="",title="a")
        Plots.plot!(pp3, collect(0:i), avals[1:i+1] ,color=:blue,label="")
        xlabel!(pp3,"iteration")
        ylabel!(pp3,"")
        xlims!(0,ITER)

        cvals = extractcomp(parsave,2)
        pp4 = Plots.plot(collect(0:i), cvals[1:i+1],seriestype=:scatter ,color=:blue,markersize=1.5,label="",title="c")
        Plots.plot!(pp4, collect(0:i), cvals[1:i+1] ,color=:blue,label="")
        xlabel!(pp4,"iteration")
        ylabel!(pp4,"")
        xlims!(0,ITER)

        γvals = extractcomp(parsave,3)
        pp5 = Plots.plot(collect(1:i), γvals[1:i],seriestype=:scatter ,color=:blue,markersize=1.5,label="",title="γ")
        Plots.plot!(pp5, collect(1:i), γvals[1:i] ,color=:blue,label="")
        xlabel!(pp5,"iteration")
        ylabel!(pp5,"")
        xlims!(0,ITER)

        l = @layout [a  b; c{0.8h} d{0.8h} e{0.8h}]
        Plots.plot(pp1,pp2,pp3,pp4,pp5, background_color = :ivory,layout=l , size = (900, 500) )
        pp1 = nothing
end


function drawobjective(objvals)
    ITER = length(objvals)
    sc2 = Plots.plot(collect(1:ITER), objvals,seriestype=:scatter ,color=:blue,markersize=1.2,label="",title="Loglikelihood approximation")
    Plots.plot!(sc2, collect(1:ITER), objvals ,color=:blue,label="")
    xlabel!(sc2,"iteration")
    ylabel!(sc2,"stoch log likelihood")
    xlims!(0,ITER)
    display(sc2)
    png(sc2,"stochlogp.png")
end

"""
    Useful for storage of a samplepath of states
    Ordering is as follows:
    1) time
    2) landmark nr
    3) for each landmark: q1, q2 p1, p2

    With m time-steps, n landmarks, this entails a vector of length m * n * 2 * d
"""
function convert_samplepath(X)
    VA = VectorOfArray(map(x->deepvec(x),X.yy))
    vec(convert(Array,VA))
end

"""
    Useful for storage of a samplepath of states
    Ordering is as follows:
    0) shape
    1) time
    2) landmark nr
    3) for each landmark: q1, q2 p1, p2

    With m time-steps, n landmarks, this entails a vector of length m * n * 2 * d
"""
function convert_samplepath(Xvec::Vector)
    nshapes = length(Xvec)
    out = [convert_samplepath(Xvec[1])]
    for k in 2:nshapes
        push!(out,convert_samplepath(Xvec[k]))
    end
    vcat(out...)
end



"""
plot initial and final shape, given by xobs0 and xobsT respectively
"""
function plotshapes(xobs0comp1,xobs0comp2,xobsTcomp1, xobsTcomp2)
    # plot initial and final shapes
    pp = Plots.plot(xobs0comp1, xobs0comp2,seriestype=:scatter, color=:black,label="q0", title="Landmark evolution")
    Plots.plot!(pp, repeat(xobs0comp1,2), repeat(xobs0comp2,2),seriestype=:path, color=:black,label="")
    Plots.plot!(pp, xobsTcomp1, xobsTcomp2,seriestype=:scatter , color=:orange,label="qT") # points move from black to orange
    Plots.plot!(pp, repeat(xobsTcomp1,2), repeat(xobsTcomp2,2),seriestype=:path, color=:orange,label="")
    pp
end
