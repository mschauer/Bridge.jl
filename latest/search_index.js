var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#Home-1",
    "page": "Home",
    "title": "Home",
    "category": "section",
    "text": ""
},

{
    "location": "index.html#Summary-1",
    "page": "Home",
    "title": "Summary",
    "category": "section",
    "text": "Stochastic calculus and univariate and multivariate stochastic processes/Markov processes in continuous time.The key objects introduced are the abstract type ContinuousTimeProcess{T} parametrised by the state space of the path, for example T == Float64 and various structs suptyping it, for example Wiener{Float64} for a real Brownian motion. These play roughly a similar role as types subtyping Distribution in the Distributions.jl package.Secondly, the struct struct SamplePath{T}\n    tt::Vector{Float64}\n    yy::Vector{T}\nendserves as container for sample path returned by direct and approximate samplers (sample, euler, ...). tt is the vector of the grid points of the simulation and yy the corresponding vector of states.Help is available at the REPL:help?> Bridge.ContinuousTimeProcess\n  ContinuousTimeProcess{T}\n\n  Types inheriting from the abstract type ContinuousTimeProcess{T}\n  characterize the properties of a T-valued stochastic process, play a similar\n  role as distribution types like Exponential in the package Distributions.Pre-defined processes defined are Wiener, WienerBridge, Gamma, LinPro (linear diffusion/generalized Ornstein-Uhlenbeck) and others."
},

{
    "location": "index.html#Features-1",
    "page": "Home",
    "title": "Features",
    "category": "section",
    "text": "Define and simulate diffusion processes in one or more dimension\nContinuous and discrete likelihood using Girsanovs theorem and transition densities\nMonte Carlo sample diffusion bridges, diffusion processes conditioned to hit a point v at a prescribed time T\nBrownian motion in one and more dimensions\nOrnstein-Uhlenbeck processes\nBessel processes\nGamma processes\nBasic stochastic calculus functionality (Ito integral, quadratic variation)\nEuler-Scheme and implicit methods (Runge-Kutta)The layout/api was originally written to be compatible with Simon Danisch\'s package FixedSizeArrays.jl. It was refactored to be compatible with StaticArrays.jl by Dan Getz.The example programs in the example/directory have additional dependencies: ConjugatePriors and a plotting library."
},

{
    "location": "manual.html#",
    "page": "Manual",
    "title": "Manual",
    "category": "page",
    "text": ""
},

{
    "location": "manual.html#Manual-1",
    "page": "Manual",
    "title": "Manual",
    "category": "section",
    "text": ""
},

{
    "location": "manual.html#Define-and-simulate-a-stochastic-process-1",
    "page": "Manual",
    "title": "Define and simulate a stochastic process",
    "category": "section",
    "text": "In this section, an Ornstein-Uhlenbeck process is defined by the stochastic differential equation    mathrmd X_t = -β mathrmdt + σ mathrmd W_tqquad(1)and a sample path is generated in three steps. β::Float64 is the mean reversion parameter  and σ::Float64 is the diffusion parameter."
},

{
    "location": "manual.html#Step-1.-Define-a-diffusion-process-OrnsteinUhlenbeck.-1",
    "page": "Manual",
    "title": "Step 1. Define a diffusion process OrnsteinUhlenbeck.",
    "category": "section",
    "text": "The new struct OrnsteinUhlenbeck is a subtype ContinuousTimeProcess{Float64} indicating that the Ornstein-Uhlenbeck process has Float64-valued trajectories.using Bridge\nstruct OrnsteinUhlenbeck  <: ContinuousTimeProcess{Float64}\n    β::Float64\n    σ::Float64 \n    function OrnsteinUhlenbeck(β::Float64, σ::Float64)\n        isnan(β) || β > 0. || error(\"Parameter β must be positive.\")\n        isnan(σ) || σ > 0. || error(\"Parameter σ must be positive.\")\n        new(β, σ)\n    end\nend\n\n# output\n"
},

{
    "location": "manual.html#Step-2.-Define-drift-and-diffusion-coefficient.-1",
    "page": "Manual",
    "title": "Step 2. Define drift and diffusion coefficient.",
    "category": "section",
    "text": "b is the dependend drift, σ the dispersion coefficient and a the diffusion coefficient. These functions expect a time t, a location x and are dispatch on the type of the process P. In this case their values are constants provided by the P argument.Bridge.b(t, x, P::OrnsteinUhlenbeck) = -P.β * x\nBridge.σ(t, x, P::OrnsteinUhlenbeck) = P.σ\nBridge.a(t, x, P::OrnsteinUhlenbeck) = P.σ^2\n\n# output\n"
},

{
    "location": "manual.html#Step-3.-Simulate-OrnsteinUhlenbeck-process-using-the-Euler-scheme.-1",
    "page": "Manual",
    "title": "Step 3. Simulate OrnsteinUhlenbeck process using the Euler scheme.",
    "category": "section",
    "text": "Generate the driving Brownian motion W of the stochastic differential equation (1) with sample. Thefirst argument is the time grid, the second arguments specifies a Float64-valued Brownian motion/Wiener process.using Random\nRandom.seed!(1)\nW = sample(0:0.1:1, Wiener())\n\n# output\n\nSamplePath{Float64}([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.0, 0.0940107, 0.214935, 0.0259463, 0.0226432, -0.24268, -0.144298, 0.581472, -0.135443, 0.0321464, 0.168574])The output is a SamplePath object assigned to W. It contains time grid W.tt and the sampled values W.yy.Generate a solution X using the Euler()-scheme, using time grid W.tt. The arguments are starting point 0.1, driving Brownian motion W and the OrnsteinUhlenbeck object with parameters β = 20.0 and σ = 1.0.X = Bridge.solve(Euler(), 0.1, W, OrnsteinUhlenbeck(20.0, 1.0));\n\n# output\n\nSamplePath{Float64}([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.1, -0.00598928, 0.126914, -0.315902, 0.312599, -0.577923, 0.676305, 0.0494658, -0.766381, 0.933971, -0.797544])This returns a SamplePath of the solution.DocTestSetup = quote\n    using Bridge\nend"
},

{
    "location": "manual.html#Tutorials-and-Notebooks-1",
    "page": "Manual",
    "title": "Tutorials and Notebooks",
    "category": "section",
    "text": "A detailed tutorial script: ./example/tutorial.jlA nice notebook detailing the generation of the logo using ordinary and stochastic differential equations (and, in fact, diffusion bridges (sic) to create a seamless loop): ./example/Bridge+Logo.ipynb"
},

{
    "location": "library.html#",
    "page": "Library",
    "title": "Library",
    "category": "page",
    "text": ""
},

{
    "location": "library.html#Library-1",
    "page": "Library",
    "title": "Library",
    "category": "section",
    "text": ""
},

{
    "location": "library.html#Bridge.ContinuousTimeProcess",
    "page": "Library",
    "title": "Bridge.ContinuousTimeProcess",
    "category": "type",
    "text": "ContinuousTimeProcess{T}\n\nTypes inheriting from the abstract type ContinuousTimeProcess{T} characterize  the properties of a T-valued stochastic process, play a similar role as distribution types like Exponential in the package Distributions.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.SamplePath",
    "page": "Library",
    "title": "Bridge.SamplePath",
    "category": "type",
    "text": "SamplePath{T} <: AbstractPath{T}\n\nThe struct\n\nstruct SamplePath{T}\n    tt::Vector{Float64}\n    yy::Vector{T}\n    SamplePath{T}(tt, yy) where {T} = new(tt, yy)\nend\n\nserves as container for discretely observed ContinuousTimeProcesses and for the sample path returned by direct and approximate samplers. tt is the vector of the grid points of the observation/simulation  and yy is the corresponding vector of states.\n\nIt supports getindex, setindex!, length, copy, vcat.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.GSamplePath",
    "page": "Library",
    "title": "Bridge.GSamplePath",
    "category": "type",
    "text": "Like VSamplePath, but with assumptions on tt and dimensionality. Planned replacement for VSamplePath\n\n\n\n\n\n"
},

{
    "location": "library.html#Base.valtype",
    "page": "Library",
    "title": "Base.valtype",
    "category": "function",
    "text": "valtype(::ContinuousTimeProcess) -> T\n\nReturns statespace (type) of a ContinuousTimeProcess{T].\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.outertype",
    "page": "Library",
    "title": "Bridge.outertype",
    "category": "function",
    "text": "outertype(P::ContinuousTimeProcess) -> T\n\nReturns the type of outer(x), where x is a state of P\n\n\n\n\n\n"
},

{
    "location": "library.html#Important-concepts-1",
    "page": "Library",
    "title": "Important concepts",
    "category": "section",
    "text": "ContinuousTimeProcess{T}\nSamplePath{T}\nBridge.GSamplePath\nvaltype\nBridge.outertype"
},

{
    "location": "library.html#Bridge.ODESolver",
    "page": "Library",
    "title": "Bridge.ODESolver",
    "category": "type",
    "text": "ODESolver\n\nAbstract (super-)type for solving methods for ordinary differential equations.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.solve!",
    "page": "Library",
    "title": "Bridge.solve!",
    "category": "function",
    "text": "solve!(method, F, X::SamplePath, x0, P) -> X, [err]\nsolve!(method, X::SamplePath, x0, F) -> X, [err]\n\nSolve ordinary differential equation (ddx) x(t) = F(t x(t)) or (ddx) x(t) = F(t x(t) P) on the fixed grid X.tt writing into X.yy .\n\nmethod::R3 - using a non-adaptive Ralston (1965) update (order 3).\n\nmethod::BS3 use non-adaptive Bogacki–Shampine method to give error estimate.\n\nCall _solve! to inline. \"Pretty fast if x is a bitstype or a StaticArray.\"\n\n\n\n\n\nsolve!(::EulerMaruyama, Y, u, W, P) -> X\n\nSolve stochastic differential equation dX_t = b(tX_t)dt + σ(tX_t)dW_t using the Euler-Maruyama scheme in place.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.solvebackward!",
    "page": "Library",
    "title": "Bridge.solvebackward!",
    "category": "function",
    "text": "Currently only timedependent sigma, as Ito correction is necessary\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.R3",
    "page": "Library",
    "title": "Bridge.R3",
    "category": "type",
    "text": "R3\n\nRalston (1965) update (order 3 step of the Bogacki–Shampine 1989 method) to solve y(t + dt) - y(t) = int_t^t+dt F(s y(s)) ds.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.BS3",
    "page": "Library",
    "title": "Bridge.BS3",
    "category": "type",
    "text": "BS3\n\nRalston (1965) update (order 3 step of the Bogacki–Shampine 1989 method) to solve y(t + dt) - y(t) = int_t^t+dt F(s y(s)) ds. Uses Bogacki–Shampine method  to give error estimate. \n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.LeftRule",
    "page": "Library",
    "title": "Bridge.LeftRule",
    "category": "type",
    "text": "LeftRule <: QuadratureRule\n\nIntegrate using left Riemann sum approximation.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.fundamental_matrix",
    "page": "Library",
    "title": "Bridge.fundamental_matrix",
    "category": "function",
    "text": "fundamental_matrix(tt, P)\n\nCompute fundamental solution.\n\n\n\n\n\n"
},

{
    "location": "library.html#Ordinary-differential-equations-and-quadrature-1",
    "page": "Library",
    "title": "Ordinary differential equations and quadrature",
    "category": "section",
    "text": "Bridge.ODESolver\nsolve!\nBridge.solvebackward!\nBridge.R3\nBridge.BS3\nLeftRule\nBridge.fundamental_matrix"
},

{
    "location": "library.html#Brownian-motion-1",
    "page": "Library",
    "title": "Brownian motion",
    "category": "section",
    "text": "Modules = [Bridge]\nPages = [\"/wiener.jl\"]"
},

{
    "location": "library.html#Bridge.a",
    "page": "Library",
    "title": "Bridge.a",
    "category": "function",
    "text": "a(t, x, P::ProcessOrCoefficients)\n\nFallback for a(t, x, P) calling σ(t, x, P)*σ(t, x, P)\'.\n\n\n\n\n\n"
},

{
    "location": "library.html#StatsBase.sample",
    "page": "Library",
    "title": "StatsBase.sample",
    "category": "function",
    "text": "sample(tt, P, x1=zero(T))\n\nSample the process P on the grid tt exactly from its transitionprob(-ability) starting in x1.\n\n\n\n\n\nsample(::Thinning, T, P::InhomogPoisson) -> tt\n\n\n\n\n\n"
},

{
    "location": "library.html#StatsBase.sample!",
    "page": "Library",
    "title": "StatsBase.sample!",
    "category": "function",
    "text": "sample!(X, P, x1=zero(T))\n\nSample the process P on the grid X.tt exactly from its transitionprob(-ability) starting in x1 writing into X.yy.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.quvar",
    "page": "Library",
    "title": "Bridge.quvar",
    "category": "function",
    "text": "quvar(X)\n\nComputes the (realized) quadratic variation of the path X.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.bracket",
    "page": "Library",
    "title": "Bridge.bracket",
    "category": "function",
    "text": "bracket(X)\nbracket(X,Y)\n\nComputes quadratic variation process of X (of X and Y).\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.ito",
    "page": "Library",
    "title": "Bridge.ito",
    "category": "function",
    "text": "ito(Y, X)\n\nIntegrate a stochastic process Y with respect to a stochastic differential dX.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.girsanov",
    "page": "Library",
    "title": "Bridge.girsanov",
    "category": "function",
    "text": "girsanov(X::SamplePath, P::ContinuousTimeProcess, Pt::ContinuousTimeProcess)\n\nGirsanov log likelihood mathrmdPmathrmdPt(X)    \n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.lp",
    "page": "Library",
    "title": "Bridge.lp",
    "category": "function",
    "text": "lp(s, x, t, y, P)\n\nLog-transition density, shorthand for logpdf(transitionprob(s,x,t,P),y).\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.llikelihood",
    "page": "Library",
    "title": "Bridge.llikelihood",
    "category": "function",
    "text": "llikelihood(X::SamplePath, P::ContinuousTimeProcess)\n\nLog-likelihood of observations X using transition density lp.\n\n\n\n\n\nllikelihood(X::SamplePath, Pº::LocalGammaProcess, P::LocalGammaProcess)\n\nLog-likelihood dPº/dP. (Up to proportionality.)\n\n\n\n\n\nllikelihood(X::SamplePath, P::LocalGammaProcess)\n\nBridge log-likelihood with respect to reference measure P.P. (Up to proportionality.)\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.solve",
    "page": "Library",
    "title": "Bridge.solve",
    "category": "function",
    "text": "solve(method::SDESolver, u, W::SamplePath, P) -> X\nsolve(method::SDESolver, u, W::SamplePath, (b, σ)) -> X\n\nSolve stochastic differential equation dX_t = b(tX_t)dt + σ(tX_t)dW_t using method in place.\n\nExample\n\nsolve(EulerMaruyama(), 1.0, sample(0:0.1:10, Wiener()), ((t,x)->-x, (t,x)->I))\n\nstruct OU <: ContinuousTimeProcess{Float64}\n    μ::Float64\nend\nBridge.b(s, x, P::OU) = -P.μ*x\nBridge.σ(s, x, P::OU) = I\n\nsolve(EulerMaruyama(), 1.0, sample(0:0.1:10, Wiener()), OU(1.4))\n\n\n\n\n\nsolve(method::SDESolver, u, W::VSamplePath, P) -> X\n\nSolve stochastic differential equation dX_t = b(tX_t)dt + σ(tX_t)dW_t using method.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.EulerMaruyama",
    "page": "Library",
    "title": "Bridge.EulerMaruyama",
    "category": "type",
    "text": "EulerMaruyama() <: SDESolver\n\nEuler-Maruyama scheme. Euler is defined as alias.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.Euler",
    "page": "Library",
    "title": "Bridge.Euler",
    "category": "type",
    "text": "EulerMaruyama() <: SDESolver\n\nEuler-Maruyama scheme. Euler is defined as alias.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.StochasticRungeKutta",
    "page": "Library",
    "title": "Bridge.StochasticRungeKutta",
    "category": "type",
    "text": "StochasticRungeKutta() <: SDESolver\n\nStochastic Runge-Kutta scheme for T<:Number-valued processes.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.StochasticHeun",
    "page": "Library",
    "title": "Bridge.StochasticHeun",
    "category": "type",
    "text": "StochasticHeun() <: SDESolver\n\nStochastic heun scheme.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.NoDrift",
    "page": "Library",
    "title": "Bridge.NoDrift",
    "category": "type",
    "text": "NoDrift(tt, B, β, a)\n\n\n\n\n\n"
},

{
    "location": "library.html#Stochastic-differential-equations-1",
    "page": "Library",
    "title": "Stochastic differential equations",
    "category": "section",
    "text": "Bridge.a\nsample\nsample!\nquvar\nbracket\nito\ngirsanov\nlp\nllikelihood\nsolve\nEulerMaruyama\nEuler\nStochasticRungeKutta\nStochasticHeun\nBridge.NoDrift"
},

{
    "location": "library.html#Bridge.R3!",
    "page": "Library",
    "title": "Bridge.R3!",
    "category": "type",
    "text": "R3!\n\nInplace Ralston (1965) update (order 3 step of the Bogacki–Shampine 1989 method) to solve y(t + dt) - y(t) = int_t^t+dt F(s y(s)) ds.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.σ!",
    "page": "Library",
    "title": "Bridge.σ!",
    "category": "function",
    "text": "σ!(t, y, Δw, tmp2, P)\n\nCompute stochastic increment at y, σ Δw, modifying tmp2.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.b!",
    "page": "Library",
    "title": "Bridge.b!",
    "category": "function",
    "text": "b!(t, y, tmp1, P)\n\nCompute drift b in y (without factor Δt, modifying tmp1.\n\n\n\n\n\n"
},

{
    "location": "library.html#In-place-solvers-1",
    "page": "Library",
    "title": "In place solvers",
    "category": "section",
    "text": "Bridge.R3!\nBridge.σ!\nBridge.b!"
},

{
    "location": "library.html#Bridge.GammaProcess",
    "page": "Library",
    "title": "Bridge.GammaProcess",
    "category": "type",
    "text": "GammaProcess\n\nA GammaProcess with jump rate γ and inverse jump size λ has increments Gamma(t*γ, 1/λ) and Levy measure\n\nν(x)=γ x^-1exp(-λ x) \n\nHere Gamma(α,θ) is the Gamma distribution in julia\'s parametrization with shape parameter α and scale θ.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.GammaBridge",
    "page": "Library",
    "title": "Bridge.GammaBridge",
    "category": "type",
    "text": "GammaBridge(t, v, P)\n\nA GammaProcess P conditional on htting v at time t.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.ExpCounting",
    "page": "Library",
    "title": "Bridge.ExpCounting",
    "category": "type",
    "text": "ExpCounting(λ)\n\nCounting process with arrival times arrival(P) = Exponential(1/λ) and unit jumps.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.CompoundPoisson",
    "page": "Library",
    "title": "Bridge.CompoundPoisson",
    "category": "type",
    "text": "CompoundPoisson{T} <: LevyProcess{T}\n\nAbstract type. For a compound Poisson process define rjumpsize(P) -> T and  arrival(P) -> Distribution.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.nu",
    "page": "Library",
    "title": "Bridge.nu",
    "category": "function",
    "text": " nu(k, P)\n\n(Bin-wise) integral of the Levy measure nu(B_k) (sic).\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.uniform_thinning!",
    "page": "Library",
    "title": "Bridge.uniform_thinning!",
    "category": "function",
    "text": "uniformthinning!(X, P::GammaProcess, γᵒ)\n\nReturn a Gamma process Y with new intensity γᵒ, such that X-Y has intensity γ-γᵒ and Y and X-Y are independent. In the limit dt to infty the new Gamma process has each of is jump removed with probability γᵒ/γ. Overwrites X with Y.\n\n\n\n\n\n"
},

{
    "location": "library.html#Levy-processes-1",
    "page": "Library",
    "title": "Levy processes",
    "category": "section",
    "text": "GammaProcess\nGammaBridge\nBridge.ExpCounting\nBridge.CompoundPoisson\nBridge.nu\nBridge.uniform_thinning!"
},

{
    "location": "library.html#Bridge.ThinningAlg",
    "page": "Library",
    "title": "Bridge.ThinningAlg",
    "category": "type",
    "text": "ThinningAlg(λmax)\n\nSampling method for InhomogPoisson by the \'thinning\' algorithm. \n\nExamples:\n\nsample(ThinningAlg(λmax), T, InhomogPoisson(λ))\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.InhomogPoisson",
    "page": "Library",
    "title": "Bridge.InhomogPoisson",
    "category": "type",
    "text": "InhomogPoisson(λ)\n\nInhomogenous Poisson process with intensity function λ(t). See also ThinningAlg.\n\n\n\n\n\n"
},

{
    "location": "library.html#Poisson-processes-1",
    "page": "Library",
    "title": "Poisson processes",
    "category": "section",
    "text": "ThinningAlg\nInhomogPoisson"
},

{
    "location": "library.html#Bridge.Bessel3Bridge",
    "page": "Library",
    "title": "Bridge.Bessel3Bridge",
    "category": "type",
    "text": "Bessel3Bridge(t, v, σ)\n\nBessel(3) bridge from below or above to the point v at time t,  not crossing v, with dispersion σ.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.BesselProp",
    "page": "Library",
    "title": "Bridge.BesselProp",
    "category": "type",
    "text": "BesselProp\n\nBessel type proposal\n\n\n\n\n\n"
},

{
    "location": "library.html#Bessel-processes-1",
    "page": "Library",
    "title": "Bessel processes",
    "category": "section",
    "text": "Bridge.Bessel3Bridge\nBridge.BesselProp"
},

{
    "location": "library.html#Bridge.endpoint!",
    "page": "Library",
    "title": "Bridge.endpoint!",
    "category": "function",
    "text": "endpoint!(X::SamplePath, v)\n\nConvenience functions setting the endpoint of X tov`.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.inner",
    "page": "Library",
    "title": "Bridge.inner",
    "category": "function",
    "text": "inner(x[, y])\n\nShort-hand for quadratic form x\'x (or x\'y).\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.cumsum0",
    "page": "Library",
    "title": "Bridge.cumsum0",
    "category": "function",
    "text": "cumsum0(x)\n\nCumulative sum starting at 0 such that cumsum0(diff(x)) ≈ x.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.mat",
    "page": "Library",
    "title": "Bridge.mat",
    "category": "function",
    "text": "mat(yy::Vector{SVector})\n\nReinterpret X or yy to an array without change in memory.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.outer",
    "page": "Library",
    "title": "Bridge.outer",
    "category": "function",
    "text": "outer(x[, y])\n\nShort-hand for quadratic form xx\' (or xy\').\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.CSpline",
    "page": "Library",
    "title": "Bridge.CSpline",
    "category": "type",
    "text": "CSpline(s, t, x, y = x, m0 = (y-x)/(t-s), m1 = (y-x)/(t-s))\n\nCubic spline parametrized by f(s) = x and f(t) = y, f(s) = m_0, f(t) = m_1.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.integrate",
    "page": "Library",
    "title": "Bridge.integrate",
    "category": "function",
    "text": "integrate(cs::CSpline, s, t)\n\nIntegrate the cubic spline from s to t.    \n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.logpdfnormal",
    "page": "Library",
    "title": "Bridge.logpdfnormal",
    "category": "function",
    "text": "logpdfnormal(x, Σ)\n\nlogpdf of centered Gaussian with covariance Σ\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.runmean",
    "page": "Library",
    "title": "Bridge.runmean",
    "category": "function",
    "text": "runmean(x)\n\nRunning mean of the vector x.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.PSD",
    "page": "Library",
    "title": "Bridge.PSD",
    "category": "type",
    "text": "PSD{T}\n\nSimple wrapper for the lower triangular Cholesky root of a positive (semi-)definite element σ.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.Gaussian",
    "page": "Library",
    "title": "Bridge.Gaussian",
    "category": "type",
    "text": "Gaussian(μ, Σ) -> P\n\nGaussian distribution with mean μand covarianceΣ. Definesrand(P)and(log-)pdf(P, x). Designed to work withNumbers,UniformScalings,StaticArraysandPSD`-matrices.\n\nImplementation details: On Σ the functions logdet, whiten and unwhiten (or cholupper as fallback for the latter two) are called.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.refine",
    "page": "Library",
    "title": "Bridge.refine",
    "category": "function",
    "text": "refine(tt, n)\n\nRefine range by decreasing stepsize by a factor n.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.quaternion",
    "page": "Library",
    "title": "Bridge.quaternion",
    "category": "function",
    "text": "quaternion(m::SMatrix{3,3})\n\nCompute the (rotation-) quarternion of a 3x3 rotation matrix. Useful to create isodensity ellipses from spheres in GL visualizations.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge._viridis",
    "page": "Library",
    "title": "Bridge._viridis",
    "category": "constant",
    "text": "_viridis\n\nColor data of the Viridis map by Nathaniel J. Smith, Stefan van Der Walt, Eric Firing from https://github.com/BIDS/colormap/blob/master/colormaps.py .\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.supnorm",
    "page": "Library",
    "title": "Bridge.supnorm",
    "category": "function",
    "text": "Supremum norm\n\n\n\n\n\n"
},

{
    "location": "library.html#Miscellaneous-1",
    "page": "Library",
    "title": "Miscellaneous",
    "category": "section",
    "text": "Bridge.endpoint!\nBridge.inner\nBridge.cumsum0\nBridge.mat\nBridge.outer\nCSpline\nBridge.integrate\nBridge.logpdfnormal\nBridge.runmean\nBridge.PSD\nBridge.Gaussian\nBridge.refine\nBridge.quaternion\nBridge._viridis\nBridge.supnorm"
},

{
    "location": "library.html#Bridge.mcstart",
    "page": "Library",
    "title": "Bridge.mcstart",
    "category": "function",
    "text": "mcstart(x) -> state\n\nCreate state for random chain online statitics. The entries/value of x are ignored\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.mcnext",
    "page": "Library",
    "title": "Bridge.mcnext",
    "category": "function",
    "text": "mcnext(state, x) -> state\n\nUpdate random chain online statistics when new chain value x was observed. Return new state.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.mcband",
    "page": "Library",
    "title": "Bridge.mcband",
    "category": "function",
    "text": "mcband(mc)\n\nCompute marginal 95% coverage interval for the chain from normal approximation.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.mcbandmean",
    "page": "Library",
    "title": "Bridge.mcbandmean",
    "category": "function",
    "text": "mcmeanband(mc)\n\nCompute marginal confidence interval for the chain mean using normal approximation\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.mcstats",
    "page": "Library",
    "title": "Bridge.mcstats",
    "category": "function",
    "text": "mcstats(mc)\n\nCompute mean and covariance estimates.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.mcmarginalstats",
    "page": "Library",
    "title": "Bridge.mcmarginalstats",
    "category": "function",
    "text": "mcmarginalstats(mcstates) -> mean, std\n\nCompute meanand marginal standard deviationsstd` for 2d plots. \n\n\n\n\n\n"
},

{
    "location": "library.html#Online-statistics-1",
    "page": "Library",
    "title": "Online statistics",
    "category": "section",
    "text": "Online updating of the tuple state = (m, m2, n) wherem - mean(x[1:n])m2 - sum of squares of differences from the current mean, textstylesum_i=1^n (x_i - bar x_n)^2n - number of iterationsmcstart\nmcnext\nmcband\nmcbandmean\nBridge.mcstats\nBridge.mcmarginalstats"
},

{
    "location": "library.html#Bridge.LinPro",
    "page": "Library",
    "title": "Bridge.LinPro",
    "category": "type",
    "text": "LinPro(B, μ::T, σ)\n\nLinear diffusion dX = B(X - μ)dt + σdW.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.Ptilde",
    "page": "Library",
    "title": "Bridge.Ptilde",
    "category": "type",
    "text": "Ptilde(cs::CSpline, σ)\n\nAffine diffusion dX = cs(t) dt + σdW  with cs a cubic spline ::CSpline.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.LinearNoiseAppr",
    "page": "Library",
    "title": "Bridge.LinearNoiseAppr",
    "category": "type",
    "text": "LinearNoiseAppr(tt, P, x, a, direction = forward)\n\nPrecursor of the linear noise approximation of P. For now no attempt is taken  to add in a linearization around the deterministic path. direction can be one of :forward, :backward or :nothing. The latter corresponds to choosing β == 0.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.LinearAppr",
    "page": "Library",
    "title": "Bridge.LinearAppr",
    "category": "type",
    "text": "LinearAppr(tt, B, β, a)\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.LinProBridge",
    "page": "Library",
    "title": "Bridge.LinProBridge",
    "category": "type",
    "text": "LinProBridge\n\nBridge process of P::LinPro with μ == 0 conditional on ending in v at time t.\n\n\n\n\n\n"
},

{
    "location": "library.html#Linear-Processes-1",
    "page": "Library",
    "title": "Linear Processes",
    "category": "section",
    "text": "LinPro\nBridge.Ptilde\nBridge.LinearNoiseAppr\nBridge.LinearAppr\nBridge.LinProBridge"
},

{
    "location": "library.html#Bridge.GuidedProp",
    "page": "Library",
    "title": "Bridge.GuidedProp",
    "category": "type",
    "text": "GuidedProp\n\nGeneral bridge proposal process, only assuming that Pt defines H and r in the right way.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.GuidedBridge",
    "page": "Library",
    "title": "Bridge.GuidedBridge",
    "category": "type",
    "text": "GuidedBridge\n\nGuided proposal process for diffusion bridge using backward recursion.\n\nGuidedBridge(tt, P, Pt, v)\n\nConstructor of guided proposal process for diffusion bridge of P to v on  the time grid tt using guiding term derived from linear process Pt.\n\nGuidedBridge(tt, P, Pt, V, H♢)\n\nGuided proposal process for diffusion bridge of P to v on  the time grid tt using guiding term derived from linear process Pt. Initialize using Bridge.gpupdate(H♢, V, L, Σ, v)\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.PartialBridge",
    "page": "Library",
    "title": "Bridge.PartialBridge",
    "category": "type",
    "text": "PartialBridge\n\nGuided proposal process for diffusion bridge using backward recursion.\n\nPartialBridge(tt, P, Pt,  L, v, Σ)\n\nGuided proposal process for a partial diffusion bridge of P to v on the time grid tt using guiding term derived from linear process Pt.\n\nSimulate with bridge!.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.PartialBridgeνH",
    "page": "Library",
    "title": "Bridge.PartialBridgeνH",
    "category": "type",
    "text": "PartialBridgeνH\n\nGuided proposal process for diffusion bridge using backward recursion.\n\nPartialBridgeνH(tt, P, Pt,  L, v,ϵ Σ)\n\nGuided proposal process for a partial diffusion bridge of P to v on the time grid tt using guiding term derived from linear process Pt.\n\nSimulate with bridge!.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.BridgePre",
    "page": "Library",
    "title": "Bridge.BridgePre",
    "category": "type",
    "text": "BridgePre() <: SDESolver\n\nPrecomputed Euler-Maruyama scheme for bridges using bi.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.BridgeProp",
    "page": "Library",
    "title": "Bridge.BridgeProp",
    "category": "type",
    "text": "BridgeProp(Target::ContinuousTimeProcess, tt, v, a, cs)\n\nSimple bridge proposal derived from a linear process with time dependent drift given by a CSpline and constant diffusion coefficient a.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.Mdb",
    "page": "Library",
    "title": "Bridge.Mdb",
    "category": "type",
    "text": "Mdb() <: SDESolver\n\nEuler scheme with the diffusion coefficient correction of the modified diffusion bridge.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.bridge",
    "page": "Library",
    "title": "Bridge.bridge",
    "category": "function",
    "text": "bridge(method, W, P) -> Y\n\nIntegrate with method, where P is a bridge proposal.\n\nExamples\n\ncs = Bridge.CSpline(tt[1], tt[end], Bridge.b(tt[1], v[1], P),  Bridge.b(tt[end], v[2], P))\nP° = BridgeProp(Pσ, v), Pσ.a, cs)\nW = sample(tt, Wiener())\nbridge(BridgePre(), W, P°)\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.bridge!",
    "page": "Library",
    "title": "Bridge.bridge!",
    "category": "function",
    "text": "bridge!(method, Y, W, P) -> Y\n\nIntegrate with method, where P is a bridge proposal overwritingY`.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.Vs",
    "page": "Library",
    "title": "Bridge.Vs",
    "category": "function",
    "text": "Vs(s, T1, T2, v, P)\n\nTime changed V for generation of U.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.gpV!",
    "page": "Library",
    "title": "Bridge.gpV!",
    "category": "function",
    "text": "gpV!(K::SamplePath, P, KT=zero(T))\n\nPrecompute V from (ddt)V = BV + β, V_T = v for a guided proposal.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.r",
    "page": "Library",
    "title": "Bridge.r",
    "category": "function",
    "text": "r(t, x, T, v, P)\n\nReturns r(tx) = operatornamegrad_x log p(tx T v) where p is the transition density of the process P.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.gpHinv!",
    "page": "Library",
    "title": "Bridge.gpHinv!",
    "category": "function",
    "text": "gpHinv!(K::SamplePath, P, KT=zero(T))\n\nPrecompute K = H^-1 from (ddt)K = BK + KB + a for a guided proposal.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.gpupdate",
    "page": "Library",
    "title": "Bridge.gpupdate",
    "category": "function",
    "text": "gpupdate(H♢, V, L, Σ, v)\ngpupdate(P, L, Σ, v)\n\nReturn updated H♢, V when observation v at time zero with error Σ is observed.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridges-1",
    "page": "Library",
    "title": "Bridges",
    "category": "section",
    "text": "GuidedProp\nBridge.GuidedBridge\nBridge.PartialBridge\nBridge.PartialBridgeνH\nBridgePre\nBridgeProp\nBridge.Mdb\nbridge\nbridge!\nBridge.Vs\nBridge.gpV!\nBridge.r\nBridge.gpHinv!\nBridge.gpupdate"
},

{
    "location": "library.html#Bridge.LocalGammaProcess",
    "page": "Library",
    "title": "Bridge.LocalGammaProcess",
    "category": "type",
    "text": "LocalGammaProcess\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.compensator0",
    "page": "Library",
    "title": "Bridge.compensator0",
    "category": "function",
    "text": "compensator0(kstart, P::LocalGammaProcess)\n\nCompensator of GammaProcess approximating the LocalGammaProcess. For kstart == 1 (only choice) this is nu_0(b_1infty).\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.compensator",
    "page": "Library",
    "title": "Bridge.compensator",
    "category": "function",
    "text": "compensator(kstart, P::LocalGammaProcess)\n\nCompensator of LocalGammaProcess  For kstart = 1, this is sum_k=1^N nu(B_k), for kstart = 0, this is sum_k=0^N nu(B_k) - C (where C is a constant).\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.θ",
    "page": "Library",
    "title": "Bridge.θ",
    "category": "function",
    "text": "θ(x, P::LocalGammaProcess)\n\nInverse jump size compared to gamma process with same alpha and beta.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.soft",
    "page": "Library",
    "title": "Bridge.soft",
    "category": "function",
    "text": "soft(t, T1, T2)\n\nTime change mapping s in [T1, T2](U-time) tot`in[T1, T2](X`-time).\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.tofs",
    "page": "Library",
    "title": "Bridge.tofs",
    "category": "function",
    "text": "tofs(s, T1, T2)\n\nTime change mapping t in [T1, T2] (X-time) to s in [T1, T2] (U-time).\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.dotVs",
    "page": "Library",
    "title": "Bridge.dotVs",
    "category": "function",
    "text": "dotVs (s, T1, T2, v, P)\n\nTime changed time derivative of V for generation of U.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.SDESolver",
    "page": "Library",
    "title": "Bridge.SDESolver",
    "category": "type",
    "text": "SDESolver\n\nAbstract (super-)type for solving methods for stochastic differential equations.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.Increments",
    "page": "Library",
    "title": "Bridge.Increments",
    "category": "type",
    "text": "Increments{S<:AbstractPath{T}}\n\nIterator over the increments of an AbstractPath.  Iterates over (i, tt[i], tt[i+1]-tt[i], yy[i+1]-y[i]).\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.sizedtype",
    "page": "Library",
    "title": "Bridge.sizedtype",
    "category": "function",
    "text": "sizedtype(x) -> T\n\nReturn an extended type which preserves size information. Makes one(T) and zero(T) for vectors possible.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.piecewise",
    "page": "Library",
    "title": "Bridge.piecewise",
    "category": "function",
    "text": "piecewise(X::SamplePath, [endtime]) -> tt, xx\n\nIf X is a jump process with piecewise constant paths and jumps in X.tt, piecewise returns coordinates path for plotting purposes. The second argument allows to choose the right endtime of the last interval.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.BridgePre!",
    "page": "Library",
    "title": "Bridge.BridgePre!",
    "category": "type",
    "text": "BridgePre!() <: SDESolver\n\nPrecomputed, replacing Euler-Maruyama scheme for bridges using bi.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.aeuler",
    "page": "Library",
    "title": "Bridge.aeuler",
    "category": "function",
    "text": "aeuler(u, s:dtmax:t, P, tau=0.5)\n\nAdaptive Euler-Maruyama scheme from https://arxiv.org/pdf/math/0601029.pdf sampling a path from u at s to t with adaptive stepsize of 2.0^(-k)*dtmax\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.MeanCov",
    "page": "Library",
    "title": "Bridge.MeanCov",
    "category": "type",
    "text": "MeanCov(itr)\n\nIterator interface for online mean and covariance Iterates are triples mean, λ, cov/λ  \n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.upsample",
    "page": "Library",
    "title": "Bridge.upsample",
    "category": "function",
    "text": "upsample(x, td, t)\n\nIf x is piecewise constant with jumps at td, return values of x at times t.\n\n\n\n\n\n"
},

{
    "location": "library.html#Bridge.viridis",
    "page": "Library",
    "title": "Bridge.viridis",
    "category": "function",
    "text": "viridis\n\nMap s onto the first maxviri viridis colors \n\n\n\n\n\n"
},

{
    "location": "library.html#Unsorted-1",
    "page": "Library",
    "title": "Unsorted",
    "category": "section",
    "text": "LocalGammaProcess\nBridge.compensator0\nBridge.compensator\nBridge.θ\nBridge.soft\nBridge.tofs\nBridge.dotVs\nBridge.SDESolver\nBridge.Increments\nBridge.sizedtype\nBridge.piecewise\nBridge.BridgePre!\nBridge.aeuler\nBridge.MeanCov\nBridge.upsample\nBridge.viridis"
},

]}
