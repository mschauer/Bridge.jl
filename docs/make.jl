using Documenter
using Bridge

makedocs(
    modules = [Bridge],
    format = :html,
    sitename = "Bridge.jl",
    authors = "Moritz Schauer and contributors",
    pages = Any[ # Compat: `Any` for 0.4 compat
        "Home" => "index.md",
        "Manual" => "manual.md",
        "Library" => "library.md",
        ],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/mschauer/Bridge.jl.git",
    julia  = "0.7",
    target = "build",
    deps = nothing,
    make = nothing,
)
