extractcomp(v,i) = map(x->x[i], v)

"""
    Plot a set of three vectors X, Y, Z, on the sphere
"""
function SpherePlot(X::T , Y::T, Z::T, 𝕊::Sphere) where {T<:AbstractArray}
    if Plots.backend() !== Plots.PlotlyBackend()
        error("Plotly() is not enabled")
    end

    R = 𝕊.R
    du = 2π/100
    dv = π/100

    u = 0.0:du:(2π+du)
    v = 0.0:dv:(π+dv)

    lenu = length(u);
    lenv = length(v);
    x = zeros(lenu, lenv); y = zeros(lenu,lenv); z = zeros(lenu,lenv)
    for i in 1:lenu
        for j in 1:lenv
            x[i,j] = R*cos.(u[i]) * sin(v[j]);
            y[i,j] = R*sin.(u[i]) * sin(v[j]);
            z[i,j] = R*cos(v[j]);
        end
    end


    Plots.surface( x,y,z,
                    axis=true,
                    alpha=0.8,
                    color = fill(RGBA(1.,1.,1.,0.8),lenu,lenv),
                    legend = false)
    Plots.plot!(X,Y,Z,
                axis = true,
                linewidth = 1.5,
                color = palette(:default)[1],
                legend = false,
                xlabel = "x",
                ylabel = "y",
                zlabel = "z")
end

# Plot a SamplePath
function SpherePlot(X::SamplePath{T}, 𝕊::Sphere) where {T}
    X1 = extractcomp(X.yy, 1)
    X2 = extractcomp(X.yy,2)
    X3 = extractcomp(X.yy,3)
    SpherePlot(X1,X2,X3, 𝕊)
end

"""
    Make a scatterplot on the sphere
"""
function SphereScatterPlot(X::T , Y::T, Z::T, 𝕊::Sphere) where {T<:AbstractArray}
    if Plots.backend() !== Plots.PlotlyBackend()
        error("Plotly() is not enabled")
    end

    R = 𝕊.R
    du = 2π/100
    dv = π/100

    u = 0.0:du:(2π+du)
    v = 0.0:dv:(π+dv)

    lenu = length(u);
    lenv = length(v);
    x = zeros(lenu, lenv); y = zeros(lenu,lenv); z = zeros(lenu,lenv)
    for i in 1:lenu
        for j in 1:lenv
            x[i,j] = R*cos.(u[i]) * sin(v[j]);
            y[i,j] = R*sin.(u[i]) * sin(v[j]);
            z[i,j] = R*cos(v[j]);
        end
    end
    Plots.plot(X,Y,Z,
                axis = false,
                seriestype = :scatter,
                color= :red,
                markersize = 2,
                legend = false,
                label = false) #, xlabel = "x", ylabel = "y", zlabel = "z")
    Plots.surface!( x,y,z,
                axis=false,
                alpha=0.8,
                color = :grey) # fill(RGBA(1.,1.,1.,0.8),lenu,lenv))
end

# with a target point
function SphereScatterPlot(X::T, Y::T, Z::T, target::T, 𝕊::Sphere) where {T<:AbstractArray}
    if Plots.backend() !== Plots.PlotlyBackend()
        error("Plotly() is not enabled")
    end
    R = 𝕊.R
    du = 2π/100
    dv = π/100

    u = 0.0:du:(2π+du)
    v = 0.0:dv:(π+dv)

    lenu = length(u);
    lenv = length(v);
    x = zeros(lenu, lenv); y = zeros(lenu,lenv); z = zeros(lenu,lenv)
    for i in 1:lenu
        for j in 1:lenv
            x[i,j] = R*cos(u[i]) * sin(v[j]);
            y[i,j] = R*sin(u[i]) * sin(v[j]);
            z[i,j] = R*cos(v[j]);
        end
    end
    Plots.plot(X,Y,Z,
                axis = true,
                seriestype = :scatter,
                color= palette(:default)[1],
                markersize = 1,
                legend = false,
                label = false,
                xlabel = "x",
                ylabel = "y",
                zlabel = "z")
    Target = Array{Float64}[]
    push!(Target, target)
    Plots.plot!(extractcomp(Target,1), extractcomp(Target,2), extractcomp(Target,3),
                seriestype = :scatter,
                color= :red,
                markersize = 2)
    Plots.surface!( x,y,z,
                axis=true,
                alpha=0.8,
                color = fill(RGBA(1.,1.,1.,0.8),lenu,lenv))
end

"""
    A plot of a trace of (for example MCMC-) updates with data and a target added
"""

function SphereFullPlot(θ, data, target, 𝕊::Sphere; PlotUpdates = true)
    if Plots.backend() !== Plots.PlotlyBackend()
        error("Plotly() is not enabled")
    end
    Target = Array{Float64}[]
    push!(Target, target)
    SpherePlot(extractcomp(θ,1), extractcomp(θ,2), extractcomp(θ,3), 𝕊)
    if PlotUpdates
        Plots.plot!(extractcomp(θ,1), extractcomp(θ,2), extractcomp(θ,3),
                    seriestype = :scatter,
                    color = palette(:default)[1],
                    markersize = 2,
                    label = "updates")
    end
    Plots.plot!(extractcomp(data,1), extractcomp(data,2), extractcomp(data,3),
                seriestype = :scatter,
                color= :black,
                markersize = 1.5,
                label = "data")
    Plots.plot!(extractcomp(Target,1), extractcomp(Target,2), extractcomp(Target,3),
                seriestype = :scatter,
                color= :red,
                markersize = 2.5,
                label = "(0,0,1)")
end
