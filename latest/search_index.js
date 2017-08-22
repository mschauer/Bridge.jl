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
    "text": "Stochastic calculus and univariate and multivariate stochastic processes/Markov processes in continuous time.The key objects introduced are the abstract type ContinuousTimeProcess{T} parametrised by the state space of the path, for example T == Float64 and various structs suptyping it, for example Wiener{Float64} for a real Brownian motion. These play roughly a similar role as types subtyping Distribution in the Distributions.jl package.Secondly, the struct struct SamplePath{T}\n    tt::Vector{Float64}\n    yy::Vector{T}\n    SamplePath{T}(tt, yy) where {T} = new(tt, yy)\nendserves as container for sample path returned by direct and approximate samplers (sample, euler, ...). tt is the vector of the grid points of the simulation and yy the corresponding vector of states.Help is available at the REPL:help?> euler\nsearch: euler euler! eulergamma default_worker_pool schedule @schedule\n\n  euler(u, W, P) -> X\n\n  Solve stochastic differential equation ``dX_t = b(t, X_t)dt + σ(t, X_t)dW_t, X_0 = u``\n  using the Euler scheme.Pre-defined processes defined are Wiener, WienerBridge, Gamma, LinPro (linear diffusion/generalized Ornstein-Uhlenbeck) and others."
},

{
    "location": "index.html#Features-1",
    "page": "Home",
    "title": "Features",
    "category": "section",
    "text": "Define and simulate diffusion processes in one or more dimension\nContinuous and discrete likelihood using Girsanovs theorem and transition densities\nMonte Carlo sample diffusion bridges, diffusion processes conditioned to hit a point v at a prescribed time T\nBrownian motion in one and more dimensions\nOrnstein-Uhlenbeck processes\nBessel processes\nGamma processes\nBasic stochastic calculus functionality (Ito integral, quadratic variation)\nEuler-Scheme and implicit methods (Runge-Kutta)The layout/api was originally written to be compatible with Simon Danisch's package FixedSizeArrays.jl. It was refactored to be compatible with StaticArrays.jl by Dan Getz.The example programs in the example/directory have additional dependencies: ConjugatePriors and a plotting library."
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
    "text": "In this section, an Ornstein-Uhlenbeck process is defined by the stochastic differential equation    mathrmd X_t = - mathrmdt +  mathrmd W_tqquad(1)and a sample path is generated in three steps. β::Float64 is the mean reversion parameter  and σ::Float64 is the diffusion parameter."
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
    "text": "Generate the driving Brownian motion W of the stochastic differential equation (1) with sample. Thefirst argument is the time grid, the second arguments specifies a Float64-valued Brownian motion/Wiener process.srand(1)\nW = sample(0:0.1:1, Wiener())\n\n# output\n\nBridge.SamplePath{Float64}([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.0, 0.0940107, 0.214935, 0.0259463, 0.0226432, -0.24268, -0.144298, 0.581472, -0.135443, 0.0321464, 0.168574])The output is a SamplePath object assigned to W. It contains time grid W.tt and the sampled values W.yy.Generate a solution X using the Euler()-scheme, using time grid W.tt. The arguments are starting point 0.1, driving Brownian motion W and the OrnsteinUhlenbeck object with parameters β = 20.0 and σ = 1.0.X = Bridge.solve(Euler(), 0.1, W, OrnsteinUhlenbeck(20.0, 1.0));\n\n# output\n\nBridge.SamplePath{Float64}([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.1, -0.00598928, 0.126914, -0.315902, 0.312599, -0.577923, 0.676305, 0.0494658, -0.766381, 0.933971, -0.797544])This returns a SamplePath of the solution.DocTestSetup = quote\n    using Bridge\nend"
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
    "category": "Type",
    "text": "ContinuousTimeProcess{T}\n\nTypes inheriting from the abstract type ContinuousTimeProcess{T} characterize  the properties of a T-valued stochastic process, play a similar role as distribution types like Exponential in the package Distributions.\n\n\n\n"
},

{
    "location": "library.html#Bridge.SamplePath",
    "page": "Library",
    "title": "Bridge.SamplePath",
    "category": "Type",
    "text": "SamplePath{T} <: AbstractPath{T}\n\nThe struct\n\nstruct SamplePath{T}\n    tt::Vector{Float64}\n    yy::Vector{T}\n    SamplePath{T}(tt, yy) where {T} = new(tt, yy)\nend\n\nserves as container for discretely observed ContinuousTimeProcesses and for the sample path returned by direct and approximate samplers. tt is the vector of the grid points of the observation/simulation  and yy is the corresponding vector of states.\n\nIt supports getindex, setindex!, length, copy, vcat, start, next, done, endof.\n\n\n\n"
},

{
    "location": "library.html#Base.valtype",
    "page": "Library",
    "title": "Base.valtype",
    "category": "Function",
    "text": "valtype(::ContinuousTimeProcess) -> T\n\nReturns statespace (type) of a ContinuousTimeProcess{T].\n\n\n\n"
},

{
    "location": "library.html#Important-concepts-1",
    "page": "Library",
    "title": "Important concepts",
    "category": "section",
    "text": "ContinuousTimeProcess{T}\nSamplePath{T}\nvaltype"
},

{
    "location": "library.html#Bridge.ODESolver",
    "page": "Library",
    "title": "Bridge.ODESolver",
    "category": "Type",
    "text": "ODESolver\n\nAbstract (super-)type for solving methods for ordinary differential equations.\n\n\n\n"
},

{
    "location": "library.html#Bridge.solve!",
    "page": "Library",
    "title": "Bridge.solve!",
    "category": "Function",
    "text": "solve!(method, F, X::SamplePath, x0, P) -> X, [err]\nsolve!(method, X::SamplePath, x0, F) -> X, [err]\n\nSolve ordinary differential equation (ddx) x(t) = F(t x(t)) or (ddx) x(t) = F(t x(t) P) on the fixed grid X.tt writing into X.yy .\n\nmethod::R3 - using a non-adaptive Ralston (1965) update (order 3).\n\nmethod::BS3 use non-adaptive Bogacki–Shampine method to give error estimate.\n\nCall _solve! to inline. \"Pretty fast if x is a bitstype or a StaticArray.\"\n\n\n\nsolve!(::EulerMaruyama, Y, u, W, P) -> X\n\nSolve stochastic differential equation dX_t = b(tX_t)dt + (tX_t)dW_t  using the Euler-Maruyama scheme in place.\n\n\n\n"
},

{
    "location": "library.html#Bridge.R3",
    "page": "Library",
    "title": "Bridge.R3",
    "category": "Type",
    "text": "R3\n\nRalston (1965) update (order 3 step of the Bogacki–Shampine 1989 method) to solve y(t + dt) - y(t) = int_t^t+dt F(s y(s)) ds.\n\n\n\n"
},

{
    "location": "library.html#Bridge.BS3",
    "page": "Library",
    "title": "Bridge.BS3",
    "category": "Type",
    "text": "BS3\n\nRalston (1965) update (order 3 step of the Bogacki–Shampine 1989 method) to solve y(t + dt) - y(t) = int_t^t+dt F(s y(s)) ds. Uses Bogacki–Shampine method  to give error estimate. \n\n\n\n"
},

{
    "location": "library.html#Bridge.LeftRule",
    "page": "Library",
    "title": "Bridge.LeftRule",
    "category": "Type",
    "text": "LeftRule <: QuadratureRule\n\nIntegrate using left Riemann sum approximation.\n\n\n\n"
},

{
    "location": "library.html#Ordinary-differential-equations-and-quadrature-1",
    "page": "Library",
    "title": "Ordinary differential equations and quadrature",
    "category": "section",
    "text": "Bridge.ODESolver\nsolve!\nBridge.R3\nBridge.BS3\nLeftRule"
},

{
    "location": "library.html#Brownian-motion-1",
    "page": "Library",
    "title": "Brownian motion",
    "category": "section",
    "text": "Modules = [Bridge]\nPages = [\"/wiener.jl\"]"
},

{
    "location": "library.html#StatsBase.sample",
    "page": "Library",
    "title": "StatsBase.sample",
    "category": "Function",
    "text": "sample(tt, P, x1=zero(T))\n\nSample the process P on the grid tt exactly from its transitionprob(-ability) starting in x1.\n\n\n\n"
},

{
    "location": "library.html#StatsBase.sample!",
    "page": "Library",
    "title": "StatsBase.sample!",
    "category": "Function",
    "text": "sample!(X, P, x1=zero(T))\n\nSample the process P on the grid X.tt exactly from its transitionprob(-ability) starting in x1 writing into X.yy.\n\n\n\n"
},

{
    "location": "library.html#Bridge.quvar",
    "page": "Library",
    "title": "Bridge.quvar",
    "category": "Function",
    "text": "quvar(X)\n\nComputes the (realized) quadratic variation of the path X.\n\n\n\n"
},

{
    "location": "library.html#Bridge.bracket",
    "page": "Library",
    "title": "Bridge.bracket",
    "category": "Function",
    "text": "bracket(X)\nbracket(X,Y)\n\nComputes quadratic variation process of X (of X and Y).\n\n\n\n"
},

{
    "location": "library.html#Bridge.ito",
    "page": "Library",
    "title": "Bridge.ito",
    "category": "Function",
    "text": "ito(Y, X)\n\nIntegrate a stochastic process Y with respect to a stochastic differential dX.\n\n\n\n"
},

{
    "location": "library.html#Bridge.girsanov",
    "page": "Library",
    "title": "Bridge.girsanov",
    "category": "Function",
    "text": "girsanov(X::SamplePath, P::ContinuousTimeProcess, Pt::ContinuousTimeProcess)\n\nGirsanov log likelihood mathrmdPmathrmdPt(X)    \n\n\n\n"
},

{
    "location": "library.html#Bridge.lp",
    "page": "Library",
    "title": "Bridge.lp",
    "category": "Function",
    "text": "lp(s, x, t, y, P)\n\nLog-transition density, shorthand for logpdf(transitionprob(s,x,t,P),y).\n\n\n\n"
},

{
    "location": "library.html#Bridge.llikelihood",
    "page": "Library",
    "title": "Bridge.llikelihood",
    "category": "Function",
    "text": "llikelihood(X::SamplePath, P::ContinuousTimeProcess)\n\nLog-likelihood of observations X using transition density lp.\n\n\n\nllikelihood(X::SamplePath, Pº::LocalGammaProcess, P::LocalGammaProcess)\n\nLog-likelihood dPº/dP. (Up to proportionality.)\n\n\n\nllikelihood(X::SamplePath, P::LocalGammaProcess)\n\nBridge log-likelihood with respect to reference measure P.P. (Up to proportionality.)\n\n\n\n"
},

{
    "location": "library.html#Bridge.solve",
    "page": "Library",
    "title": "Bridge.solve",
    "category": "Function",
    "text": "solve!(method::SDESolver, Y, u, W::SamplePath, P) -> X\n\nSolve stochastic differential equation dX_t = b(tX_t)dt + (tX_t)dW_t  using method in place.\n\n\n\nsolve!(method::SDESolver, Y, u, W::VSamplePath, P) -> X\n\nSolve stochastic differential equation dX_t = b(tX_t)dt + (tX_t)dW_t  using method in place.\n\n\n\n"
},

{
    "location": "library.html#Bridge.euler",
    "page": "Library",
    "title": "Bridge.euler",
    "category": "Function",
    "text": "euler(u, W, P) -> X\n\nSolve stochastic differential equation dX_t = b(tX_t)dt + (tX_t)dW_t using the Euler scheme.\n\n\n\n"
},

{
    "location": "library.html#Bridge.euler!",
    "page": "Library",
    "title": "Bridge.euler!",
    "category": "Function",
    "text": "euler!(Y, u, W, P) -> X\n\nSolve stochastic differential equation dX_t = b(tX_t)dt + (tX_t)dW_t  using the Euler scheme in place.\n\n\n\n"
},

{
    "location": "library.html#Bridge.EulerMaruyama",
    "page": "Library",
    "title": "Bridge.EulerMaruyama",
    "category": "Type",
    "text": "EulerMaruyama() <: SDESolver\n\nEuler-Maruyama scheme. Euler is defined as alias.\n\n\n\n"
},

{
    "location": "library.html#Bridge.Euler",
    "page": "Library",
    "title": "Bridge.Euler",
    "category": "Type",
    "text": "EulerMaruyama() <: SDESolver\n\nEuler-Maruyama scheme. Euler is defined as alias.\n\n\n\n"
},

{
    "location": "library.html#Bridge.StochasticRungeKutta",
    "page": "Library",
    "title": "Bridge.StochasticRungeKutta",
    "category": "Type",
    "text": "StochasticRungeKutta() <: SDESolver\n\nStochastic Runge-Kutta scheme for T<:Number-valued processes.\n\n\n\n"
},

{
    "location": "library.html#Bridge.StochasticHeun",
    "page": "Library",
    "title": "Bridge.StochasticHeun",
    "category": "Type",
    "text": "StochasticHeun() <: SDESolver\n\nStochastic heun scheme.\n\n\n\n"
},

{
    "location": "library.html#Stochastic-differential-equations-1",
    "page": "Library",
    "title": "Stochastic differential equations",
    "category": "section",
    "text": "sample\nsample!\nquvar\nbracket\nito\ngirsanov\nlp\nllikelihood\nsolve\neuler\neuler!\nEulerMaruyama\nEuler\nStochasticRungeKutta\nStochasticHeun"
},

{
    "location": "library.html#Bridge.GammaProcess",
    "page": "Library",
    "title": "Bridge.GammaProcess",
    "category": "Type",
    "text": "GammaProcess\n\nA GammaProcess with jump rate γ and inverse jump size λ has increments Gamma(t*γ, 1/λ) and Levy measure\n\n(x)= x^-1exp(- x) \n\nHere Gamma(α,θ) is the Gamma distribution in julia's parametrization with shape parameter α and scale θ.\n\n\n\n"
},

{
    "location": "library.html#Bridge.GammaBridge",
    "page": "Library",
    "title": "Bridge.GammaBridge",
    "category": "Type",
    "text": "GammaBridge(t, v, P)\n\nA GammaProcess P conditional on htting v at time t.\n\n\n\n"
},

{
    "location": "library.html#Bridge.ExpCounting",
    "page": "Library",
    "title": "Bridge.ExpCounting",
    "category": "Type",
    "text": "ExpCounting(λ)\n\nCounting process with arrival times arrival(P) = Exponential(1/λ) and unit jumps.\n\n\n\n"
},

{
    "location": "library.html#Bridge.CompoundPoisson",
    "page": "Library",
    "title": "Bridge.CompoundPoisson",
    "category": "Type",
    "text": "CompoundPoisson{T} <: LevyProcess{T}\n\nAbstract type. For a compound Poisson process define rjumpsize(P) -> T and  arrival(P) -> Distribution.\n\n\n\n"
},

{
    "location": "library.html#Bridge.nu",
    "page": "Library",
    "title": "Bridge.nu",
    "category": "Function",
    "text": " nu(k,P)\n\n(Bin-wise) integral of the Levy measure nu(B_k).\n\n\n\n"
},

{
    "location": "library.html#Levy-processes-1",
    "page": "Library",
    "title": "Levy processes",
    "category": "section",
    "text": "GammaProcess\nGammaBridge\nBridge.ExpCounting\nBridge.CompoundPoisson\nBridge.nu "
},

{
    "location": "library.html#Bridge.endpoint!",
    "page": "Library",
    "title": "Bridge.endpoint!",
    "category": "Function",
    "text": "endpoint!(X::SamplePath, v)\n\nConvenience functions setting the endpoint of X tov`.\n\n\n\n"
},

{
    "location": "library.html#Bridge.inner",
    "page": "Library",
    "title": "Bridge.inner",
    "category": "Function",
    "text": "inner(x[, y])\n\nShort-hand for quadratic form x'x (or x'y).\n\n\n\n"
},

{
    "location": "library.html#Bridge.cumsum0",
    "page": "Library",
    "title": "Bridge.cumsum0",
    "category": "Function",
    "text": "cumsum0(x)\n\nCumulative sum starting at 0 such that cumsum0(diff(x)) ≈ x.\n\n\n\n"
},

{
    "location": "library.html#Bridge.mat",
    "page": "Library",
    "title": "Bridge.mat",
    "category": "Function",
    "text": "mat(yy::Vector{SVector})\n\nReinterpret X or yy to an array without change in memory.\n\n\n\n"
},

{
    "location": "library.html#Bridge.outer",
    "page": "Library",
    "title": "Bridge.outer",
    "category": "Function",
    "text": "outer(x[, y])\n\nShort-hand for quadratic form xx' (or xy').\n\n\n\n"
},

{
    "location": "library.html#Bridge.CSpline",
    "page": "Library",
    "title": "Bridge.CSpline",
    "category": "Type",
    "text": "CSpline(s, t, x, y = x, m0 = (y-x)/(t-s), m1 = (y-x)/(t-s))\n\nCubic spline parametrized by f(s) = x and f(t) = y, f(s) = m_0, f(t) = m_1.\n\n\n\n"
},

{
    "location": "library.html#Bridge.integrate",
    "page": "Library",
    "title": "Bridge.integrate",
    "category": "Function",
    "text": "integrate(cs::CSpline, s, t)\n\nIntegrate the cubic spline from s to t.    \n\n\n\n"
},

{
    "location": "library.html#Bridge.logpdfnormal",
    "page": "Library",
    "title": "Bridge.logpdfnormal",
    "category": "Function",
    "text": "logpdfnormal(x, A)\n\nlogpdf of centered gaussian with covariance A\n\n\n\n"
},

{
    "location": "library.html#Miscellaneous-1",
    "page": "Library",
    "title": "Miscellaneous",
    "category": "section",
    "text": "Bridge.endpoint!\nBridge.inner\nBridge.cumsum0\nBridge.mat\nBridge.outer\nCSpline\nBridge.integrate \nBridge.logpdfnormal"
},

{
    "location": "library.html#Bridge.mcstart",
    "page": "Library",
    "title": "Bridge.mcstart",
    "category": "Function",
    "text": "mcstart(x) -> state\n\nCreate state for random chain online statitics. The entries/value of x are ignored\n\n\n\n"
},

{
    "location": "library.html#Bridge.mcnext",
    "page": "Library",
    "title": "Bridge.mcnext",
    "category": "Function",
    "text": "mcnext(state, x) -> state\n\nUpdate random chain online statistics when new chain value x was observed. Return new state.\n\n\n\n"
},

{
    "location": "library.html#Bridge.mcband",
    "page": "Library",
    "title": "Bridge.mcband",
    "category": "Function",
    "text": "mcband(mc)\n\nCompute marginal 95% coverage interval for the chain from normal approximation.\n\n\n\n"
},

{
    "location": "library.html#Bridge.mcbandmean",
    "page": "Library",
    "title": "Bridge.mcbandmean",
    "category": "Function",
    "text": "mcmeanband(mc)\n\nCompute marginal confidence interval for the chain mean using normal approximation\n\n\n\n"
},

{
    "location": "library.html#Online-statistics-1",
    "page": "Library",
    "title": "Online statistics",
    "category": "section",
    "text": "Online updating of the tuple state = (m, m2, n) wherem - mean(x[1:n])m2 - sum of squares of differences from the current mean, textstylesum_i=1^n (x_i - bar x_n)^2n - number of iterationsmcstart\nmcnext\nmcband\nmcbandmean"
},

{
    "location": "library.html#Bridge.LinPro",
    "page": "Library",
    "title": "Bridge.LinPro",
    "category": "Type",
    "text": "LinPro(B, μ::T, σ)\n\nLinear diffusion dX = B(X - )dt + dW.\n\n\n\n"
},

{
    "location": "library.html#Bridge.Ptilde",
    "page": "Library",
    "title": "Bridge.Ptilde",
    "category": "Type",
    "text": "Ptilde(cs::CSpline, σ)\n\nAffine diffusion dX = cs(t) dt + dW  with cs a cubic spline ::CSpline.\n\n\n\n"
},

{
    "location": "library.html#Linear-Processes-1",
    "page": "Library",
    "title": "Linear Processes",
    "category": "section",
    "text": "LinPro\nBridge.Ptilde"
},

{
    "location": "library.html#Bridge.GuidedProp",
    "page": "Library",
    "title": "Bridge.GuidedProp",
    "category": "Type",
    "text": "GuidedProp\n\nGeneral bridge proposal process\n\n\n\n"
},

{
    "location": "library.html#Bridge.GuidedBridge",
    "page": "Library",
    "title": "Bridge.GuidedBridge",
    "category": "Type",
    "text": "GuidedBridge\n\nGuided proposal process for diffusion bridge.\n\n\n\n"
},

{
    "location": "library.html#Bridge.Mdb",
    "page": "Library",
    "title": "Bridge.Mdb",
    "category": "Type",
    "text": "Mdb() <: SDESolver\n\nEuler scheme with the diffusion coefficient correction of the modified diffusion bridge.\n\n\n\n"
},

{
    "location": "library.html#Bridge.bridge",
    "page": "Library",
    "title": "Bridge.bridge",
    "category": "Function",
    "text": "bridge(method, W, P) -> Y\n\nIntegrate with method, where `P is a bridge proposal.\n\n\n\n"
},

{
    "location": "library.html#Bridge.bridge!",
    "page": "Library",
    "title": "Bridge.bridge!",
    "category": "Function",
    "text": "bridge!(method, Y, W, P) -> Y\n\nIntegrate with method, where P is a bridge proposal overwritingY`.\n\n\n\n"
},

{
    "location": "library.html#Bridge.Vs",
    "page": "Library",
    "title": "Bridge.Vs",
    "category": "Function",
    "text": "Vs(s, T1, T2, v, P)\n\nTime changed V for generation of U.\n\n\n\n"
},

{
    "location": "library.html#Bridge.r",
    "page": "Library",
    "title": "Bridge.r",
    "category": "Function",
    "text": "r(t, x, T, v, P)\n\nReturns r(tx) = operatornamegrad_x log p(tx T v) where p is the transition density of the process P.\n\n\n\n"
},

{
    "location": "library.html#Bridge.gpK!",
    "page": "Library",
    "title": "Bridge.gpK!",
    "category": "Function",
    "text": "gpK!(K::SamplePath, P)\n\nPrecompute K = H^-1 from (ddt)K = BK + KB + a for a guided proposal.\n\n\n\n"
},

{
    "location": "library.html#Bridges-1",
    "page": "Library",
    "title": "Bridges",
    "category": "section",
    "text": "GuidedProp\nBridge.GuidedBridge\nBridge.Mdb\nbridge\nbridge!\nBridge.Vs\nBridge.r\nBridge.gpK!"
},

{
    "location": "library.html#Bridge.LocalGammaProcess",
    "page": "Library",
    "title": "Bridge.LocalGammaProcess",
    "category": "Type",
    "text": "LocalGammaProcess\n\n\n\n"
},

{
    "location": "library.html#Bridge.compensator0",
    "page": "Library",
    "title": "Bridge.compensator0",
    "category": "Function",
    "text": "compensator0(kstart, P::LocalGammaProcess)\n\nCompensator of GammaProcess approximating the LocalGammaProcess. For kstart == 1 (only choice) this is nu_0(b_1infty).\n\n\n\n"
},

{
    "location": "library.html#Bridge.compensator",
    "page": "Library",
    "title": "Bridge.compensator",
    "category": "Function",
    "text": "compensator(kstart, P::LocalGammaProcess)\n\nCompensator of LocalGammaProcess  For kstart = 1, this is sum_k=1^N nu(B_k), for kstart = 0, this is sum_k=0^N nu(B_k) - C (where C is a constant).\n\n\n\n"
},

{
    "location": "library.html#Bridge.θ",
    "page": "Library",
    "title": "Bridge.θ",
    "category": "Function",
    "text": "θ(x, P::LocalGammaProcess)\n\nInverse jump size compared to gamma process with same alpha and beta.\n\n\n\n"
},

{
    "location": "library.html#Bridge.soft",
    "page": "Library",
    "title": "Bridge.soft",
    "category": "Function",
    "text": "soft(t, T1, T2)\n\nTime change mapping s in [T1, T2](U-time) tot`in[T1, T2](X`-time).\n\n\n\n"
},

{
    "location": "library.html#Bridge.tofs",
    "page": "Library",
    "title": "Bridge.tofs",
    "category": "Function",
    "text": "tofs(s, T1, T2)\n\nTime change mapping t in [T1, T2] (X-time) to s in [T1, T2] (U-time).\n\n\n\n"
},

{
    "location": "library.html#Bridge.dotVs",
    "page": "Library",
    "title": "Bridge.dotVs",
    "category": "Function",
    "text": "dotVs (s, T1, T2, v, P)\n\nTime changed time derivative of V for generation of U.\n\n\n\n"
},

{
    "location": "library.html#Bridge.SDESolver",
    "page": "Library",
    "title": "Bridge.SDESolver",
    "category": "Type",
    "text": "SDESolver\n\nAbstract (super-)type for solving methods for stochastic differential equations.\n\n\n\n"
},

{
    "location": "library.html#Bridge.Increments",
    "page": "Library",
    "title": "Bridge.Increments",
    "category": "Type",
    "text": "Increments{S<:AbstractPath{T}}\n\nIterator over the increments of an AbstractPath.  Iterates over (i, tt[i], tt[i+1]-tt[i], yy[i+1]-y[i]).\n\n\n\n"
},

{
    "location": "library.html#Unsorted-1",
    "page": "Library",
    "title": "Unsorted",
    "category": "section",
    "text": "LocalGammaProcess\nBridge.compensator0 \nBridge.compensator\nBridge.θ \nBridge.soft\nBridge.tofs\nBridge.dotVs\nBridge.SDESolver\nBridge.Increments"
},

]}
