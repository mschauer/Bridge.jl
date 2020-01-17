extractcomp(v,i) = map(x->x[i], v)

"""
    Plot a set of three vectors X, Y, Z, on the Torus
"""
function TorusPlot(X::T, Y::T, Z::T, 𝕋::Torus) where {T<:AbstractArray}
    if Plots.backend() !== Plots.PlotlyBackend()
        error("Plotly() is not enabled")
    end
    n = 100
    ϑ = [0;2*(0.5:n-0.5)/n;2]
    φ = [0;2*(0.5:n-0.5)/n;2]
    x = [(𝕋.R+𝕋.r*cospi(φ))*cospi(ϑ) for ϑ in ϑ, φ in φ]
    y = [(𝕋.R+𝕋.r*cospi(φ))*sinpi(ϑ) for ϑ in ϑ, φ in φ]
    z = [𝕋.r*sinpi(φ) for ϑ in ϑ, φ in φ]

    lenϑ = length(ϑ)
    lenφ = length(φ)
    rng = 𝕋.R+𝕋.r
    # Set plots
    Plots.surface(x,y,z,
                    axis=true,
                    alpha=0.5,
                    legend = false,
                    color = :grey, #fill(RGBA(1.,1.,1.,0.8),lenu,lenv),
                    xlim = (-rng-1, rng+1),
                    ylim = (-rng-1, rng+1),
                    zlim = (-rng-1, rng+1)
                    )
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
function TorusPlot(X::SamplePath{T}, 𝕋::Torus) where {T}
    X1 = extractcomp(X.yy,1)
    X2 = extractcomp(X.yy,2)
    X3 = extractcomp(X.yy,3)
    TorusPlot(X1, X2, X3, 𝕋)
end

"""
    Make a scatterplot on the Torus
"""
function TorusScatterPlot(X::T, Y::T, Z::T, 𝕋::Torus) where {T<:AbstractArray}
    if Plots.backend() !== Plots.PlotlyBackend()
        error("Plotly() is not enabled")
    end

    rng = 𝕋.R+𝕋.r
    n = 100
    ϑ = [0;2*(0.5:n-0.5)/n;2]
    φ = [0;2*(0.5:n-0.5)/n;2]
    x = [(𝕋.R+𝕋.r*cospi(φ))*cospi(ϑ) for ϑ in ϑ, φ in φ]
    y = [(𝕋.R+𝕋.r*cospi(φ))*sinpi(ϑ) for ϑ in ϑ, φ in φ]
    z = [𝕋.r*sinpi(φ) for ϑ in ϑ, φ in φ]

    lenϑ = length(ϑ)
    lenφ = length(φ)
    Plots.surface(x,y,z,
                    axis=true,
                    alpha=0.5,
                    legend = false,
                    color = :grey,
                    xlim = (-rng-1, rng+1),
                    ylim = (-rng-1, rng+1),
                    zlim = (-rng-1, rng+1)
                    )
    Plots.plot!(X,Y,Z,
                    axis = true,
                    seriestype = :scatter,
                    color= palette(:default)[1],
                    markersize = 1,
                    legend = false,
                    label = false,
                    xlabel = "x",
                    ylabel = "y",
                    zlabel = "z")
end

# with a target point
function TorusScatterPlot(X::T, Y::T, Z::T, target::T, 𝕋::Torus) where {T<:AbstractArray}
    if Plots.backend() !== Plots.PlotlyBackend()
        error("Plotly() is not enabled")
    end
    TorusScatterPlot(X, Y, Z, 𝕋)
    Target = Array{Float64}[]
    push!(Target, target)
    Plots.plot!(extractcomp(Target,1), extractcomp(Target,2), extractcomp(Target,3),
                seriestype = :scatter,
                color= :red,
                markersize = 2)
end

"""
    A plot of a trace of (for example MCMC-) updates with data and a target added
"""
function TorusFullPlot(θ, data, target, 𝕋; PlotUpdates = true)
    if Plots.backend() !== Plots.PlotlyBackend()
        error("Plotly() is not enabled")
    end
    Target = Array{Float64}[]
    push!(Target, target)
    TorusPlot(extractcomp(θ,1), extractcomp(θ,2), extractcomp(θ,3), 𝕋)
    if PlotUpdates
        Plots.plot!(extractcomp(θ,1), extractcomp(θ,2), extractcomp(θ,3),
                    seriestype = :scatter,
                    color = :yellow,
                    markersize = 2,
                    label = "Updates")
    end
    Plots.plot!(extractcomp(data,1), extractcomp(data,2), extractcomp(data,3),
                seriestype = :scatter,
                color= :black,
                markersize = 1.5,
                label = "Data")
    Plots.plot!(extractcomp(Target,1), extractcomp(Target,2), extractcomp(Target,3),
                seriestype = :scatter,
                color= :red,
                markersize = 2.5,
                label = "Target")
end
