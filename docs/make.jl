using Documenter
using Bridge

makedocs(
    modules = [Bridge],
    format = :html,
    sitename = "Bridge",
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/mschauer/Bridge.jl.git"
)
