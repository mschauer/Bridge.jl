var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Bridge.jl",
    "title": "Bridge.jl",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#Bridge.jl-1",
    "page": "Bridge.jl",
    "title": "Bridge.jl",
    "category": "section",
    "text": "Documentation for Bridge.jl"
},

{
    "location": "index.html#Bridge.ContinuousTimeProcess",
    "page": "Bridge.jl",
    "title": "Bridge.ContinuousTimeProcess",
    "category": "Type",
    "text": "ContinuousTimeProcess{T}\n\nTypes inheriting from the abstract type ContinuousTimeProcess{T} characterize  the properties of a T-valued stochastic process, play a similar role as distribution types like Exponential in the package Distributions.\n\n\n\n"
},

{
    "location": "index.html#Bridge.SamplePath",
    "page": "Bridge.jl",
    "title": "Bridge.SamplePath",
    "category": "Type",
    "text": "SamplePath{T} <: AbstractPath{T}\n\nThe struct\n\nstruct SamplePath{T}\n    tt::Vector{Float64}\n    yy::Vector{T}\n    SamplePath{T}(tt, yy) where {T} = new(tt, yy)\nend\n\nserves as container for discretely observed ContinuousTimeProcesses and for the sample path returned by direct and approximate samplers. tt is the vector of the grid points of the observation/simulation  and yy is the corresponding vector of states.\n\nIt supports getindex, setindex!, length, copy, vcat, start, next, done, endof.\n\n\n\n"
},

{
    "location": "index.html#Base.valtype",
    "page": "Bridge.jl",
    "title": "Base.valtype",
    "category": "Function",
    "text": "valtype(::ContinuousTimeProcess) -> T\n\nReturns statespace (type) of a ContinuousTimeProcess{T].\n\n\n\n"
},

{
    "location": "index.html#Important-concepts-1",
    "page": "Bridge.jl",
    "title": "Important concepts",
    "category": "section",
    "text": "ContinuousTimeProcess{T}\nSamplePath{T}\nvaltype"
},

{
    "location": "index.html#Bridge.solve!",
    "page": "Bridge.jl",
    "title": "Bridge.solve!",
    "category": "Function",
    "text": "solve!(method, X::SamplePath, x0, P) -> X, [err]\n\nSolve ordinary differential equation (ddx) x(t) = F(t x(t) P) on the fixed grid X.tt writing into X.yy \n\nmethod::R3 - using a non-adaptive Ralston (1965) update (order 3).\n\nmethod::BS3 use non-adaptive Bogacki–Shampine method to give error estimate.\n\nCall _solve! to inline. \"Pretty fast if x is a bitstype or a StaticArray.\"\n\n\n\n"
},

{
    "location": "index.html#Bridge.R3",
    "page": "Bridge.jl",
    "title": "Bridge.R3",
    "category": "Type",
    "text": "R3\n\nRalston (1965) update (order 3 step of the Bogacki–Shampine 1989 method) to solve y(t + dt) - y(t) = int_t^t+dt f(s y(s)) ds.\n\n\n\n"
},

{
    "location": "index.html#Bridge.BS3",
    "page": "Bridge.jl",
    "title": "Bridge.BS3",
    "category": "Type",
    "text": "BS3\n\nRalston (1965) update (order 3 step of the Bogacki–Shampine 1989 method) to solve y(t + dt) - y(t) = int_t^t+dt f(s y(s)) ds. Uses Bogacki–Shampine method  to give error estimate. \n\n\n\n"
},

{
    "location": "index.html#Ordinary-differential-equations-1",
    "page": "Bridge.jl",
    "title": "Ordinary differential equations",
    "category": "section",
    "text": "solve!\nBridge.R3\nBridge.BS3"
},

{
    "location": "index.html#Brownian-motion-1",
    "page": "Bridge.jl",
    "title": "Brownian motion",
    "category": "section",
    "text": "Modules = [Bridge]\nPages = [\"/wiener.jl\"]"
},

{
    "location": "index.html#StatsBase.sample",
    "page": "Bridge.jl",
    "title": "StatsBase.sample",
    "category": "Function",
    "text": "sample(tt, P, x1=zero(T))\n\nSample the process P on the grid tt exactly from its transitionprob(-ability) starting in x1.\n\n\n\n"
},

{
    "location": "index.html#StatsBase.sample!",
    "page": "Bridge.jl",
    "title": "StatsBase.sample!",
    "category": "Function",
    "text": "sample!(X, P, x1=zero(T))\n\nSample the process P on the grid X.tt exactly from its transitionprob(-ability) starting in x1 writing into X.yy.\n\n\n\n"
},

{
    "location": "index.html#Bridge.quvar",
    "page": "Bridge.jl",
    "title": "Bridge.quvar",
    "category": "Function",
    "text": "quvar(X)\n\nComputes quadratic variation of X.\n\n\n\n"
},

{
    "location": "index.html#Bridge.bracket",
    "page": "Bridge.jl",
    "title": "Bridge.bracket",
    "category": "Function",
    "text": "bracket(X)\nbracket(X,Y)\n\nComputes quadratic variation process of X (of X and Y).\n\n\n\n"
},

{
    "location": "index.html#Bridge.ito",
    "page": "Bridge.jl",
    "title": "Bridge.ito",
    "category": "Function",
    "text": "ito(Y, X)\n\nIntegrate a valued stochastic process with respect to a stochastic differential.\n\n\n\n"
},

{
    "location": "index.html#Bridge.girsanov",
    "page": "Bridge.jl",
    "title": "Bridge.girsanov",
    "category": "Function",
    "text": "girsanov(X::SamplePath, P::ContinuousTimeProcess, Pt::ContinuousTimeProcess)\n\nGirsanov log likelihood mathrmdPmathrmdPt(X)    \n\n\n\n"
},

{
    "location": "index.html#Bridge.lp",
    "page": "Bridge.jl",
    "title": "Bridge.lp",
    "category": "Function",
    "text": "lp(s, x, t, y, P)\n\nLog-transition density, shorthand for logpdf(transitionprob(s,x,t,P),y).\n\n\n\n"
},

{
    "location": "index.html#Bridge.llikelihood",
    "page": "Bridge.jl",
    "title": "Bridge.llikelihood",
    "category": "Function",
    "text": "llikelihood(X::SamplePath, P::ContinuousTimeProcess)\n\nLog-likelihood of observations X using transition density lp.\n\n\n\nllikelihood(X::SamplePath, Pº::LocalGammaProcess, P::LocalGammaProcess)\n\nLog-likelihood dPº/dP. (Up to proportionality.)\n\n\n\nllikelihood(X::SamplePath, P::LocalGammaProcess)\n\nBridge log-likelihood with respect to reference measure P.P. (Up to proportionality.)\n\n\n\n"
},

{
    "location": "index.html#Stochastic-differential-equations-1",
    "page": "Bridge.jl",
    "title": "Stochastic differential equations",
    "category": "section",
    "text": "sample\nsample!\nquvar\nbracket\nito\ngirsanov\nlp\nllikelihood"
},

{
    "location": "index.html#Bridge.endpoint!",
    "page": "Bridge.jl",
    "title": "Bridge.endpoint!",
    "category": "Function",
    "text": "endpoint!(X::SamplePath, v)\n\nConvenience functions setting the endpoint of X tov`.\n\n\n\n"
},

{
    "location": "index.html#Miscellaneous-1",
    "page": "Bridge.jl",
    "title": "Miscellaneous",
    "category": "section",
    "text": "Bridge.endpoint!"
},

{
    "location": "index.html#Bridge.ODESolver",
    "page": "Bridge.jl",
    "title": "Bridge.ODESolver",
    "category": "Type",
    "text": "ODESolver\n\nAbstract (super-)type for solving methods for ordinary differential equations.\n\n\n\n"
},

{
    "location": "index.html#Bridge.mcbandmean",
    "page": "Bridge.jl",
    "title": "Bridge.mcbandmean",
    "category": "Function",
    "text": "mcmeanband(mc)\n\nCompute marginal confidence interval for the chain mean using normal approximation\n\n\n\n"
},

{
    "location": "index.html#Bridge.cumsum0",
    "page": "Bridge.jl",
    "title": "Bridge.cumsum0",
    "category": "Function",
    "text": "cumsum0\n\nCumulative sum starting at 0, \n\n\n\n"
},

{
    "location": "index.html#Bridge.LocalGammaProcess",
    "page": "Bridge.jl",
    "title": "Bridge.LocalGammaProcess",
    "category": "Type",
    "text": "LocalGammaProcess\n\n\n\n"
},

{
    "location": "index.html#Bridge.Ptilde",
    "page": "Bridge.jl",
    "title": "Bridge.Ptilde",
    "category": "Type",
    "text": "Ptilde(cs::CSpline, σ)\n\nAffine diffusion dX = cs(t) dt + dW  with cs a \n\n\n\n"
},

{
    "location": "index.html#Bridge.GuidedProp",
    "page": "Bridge.jl",
    "title": "Bridge.GuidedProp",
    "category": "Type",
    "text": "GuidedProp\n\nGeneral bridge proposal process\n\n\n\n"
},

{
    "location": "index.html#Bridge.compensator0",
    "page": "Bridge.jl",
    "title": "Bridge.compensator0",
    "category": "Function",
    "text": "compensator0(kstart, P::LocalGammaProcess)\n\nCompensator of GammaProcess approximating the LocalGammaProcess. For kstart == 1 (only choice) this is nu_0(b_1infty).\n\n\n\n"
},

{
    "location": "index.html#Bridge.mcstart",
    "page": "Bridge.jl",
    "title": "Bridge.mcstart",
    "category": "Function",
    "text": "mcstart(x) -> state\n\nCreate state for random chain online statitics. The entries/value of x are ignored\n\n\n\n"
},

{
    "location": "index.html#Bridge.CSpline",
    "page": "Bridge.jl",
    "title": "Bridge.CSpline",
    "category": "Type",
    "text": "CSpline(s, t, x, y = x, m0 = (y-x)/(t-s), m1 = (y-x)/(t-s))\n\nCubic spline parametrized by f(s) = x and f(t) = y, f(s) = m0, f(t) = m1.\n\n\n\n"
},

{
    "location": "index.html#Bridge.tofs",
    "page": "Bridge.jl",
    "title": "Bridge.tofs",
    "category": "Function",
    "text": "tofs(s, T1, T2)\n\nTime change mapping t in [T1, T2] (X-time) to s in [T1, T2] (U-time).\n\n\n\n"
},

{
    "location": "index.html#Bridge.r",
    "page": "Bridge.jl",
    "title": "Bridge.r",
    "category": "Function",
    "text": "r(t, x, T, v, P)\n\nReturns r(tx) = operatornamegrad_x log p(tx T v) where p is the transition density of the process P.\n\n\n\n"
},

{
    "location": "index.html#Bridge.θ",
    "page": "Bridge.jl",
    "title": "Bridge.θ",
    "category": "Function",
    "text": "θ(x, P::LocalGammaProcess)\n\nInverse jump size compared to gamma process with same alpha and beta.\n\n\n\n"
},

{
    "location": "index.html#Bridge.gpK!",
    "page": "Bridge.jl",
    "title": "Bridge.gpK!",
    "category": "Function",
    "text": "gpK!(K::SamplePath, P)\n\nPrecompute K = H^-1 from (ddt)K = BK + KB + a for a guided proposal.\n\n\n\n"
},

{
    "location": "index.html#Bridge.mcnext",
    "page": "Bridge.jl",
    "title": "Bridge.mcnext",
    "category": "Function",
    "text": "mcnext(state, x) -> state\n\nUpdate random chain online statistics when new chain value x was observed. Return new state.\n\n\n\n"
},

{
    "location": "index.html#Bridge.compensator",
    "page": "Bridge.jl",
    "title": "Bridge.compensator",
    "category": "Function",
    "text": "compensator(kstart, P::LocalGammaProcess)\n\nCompensator of LocalGammaProcess  For kstart = 1, this is sum_k=1^N nu(B_k), for kstart = 0, this is sum_k=0^N nu(B_k) - C (where C is a constant).\n\n\n\n"
},

{
    "location": "index.html#Bridge.outer",
    "page": "Bridge.jl",
    "title": "Bridge.outer",
    "category": "Function",
    "text": "outer(x[, y])\n\nShort-hand for quadratic form xx' (or xy').\n\n\n\n"
},

{
    "location": "index.html#Bridge.integrate",
    "page": "Bridge.jl",
    "title": "Bridge.integrate",
    "category": "Function",
    "text": "integrate(cs::CSpline, s, t)\n\nIntegrate the cubic spline from s to t.    \n\n\n\n"
},

{
    "location": "index.html#Bridge.soft",
    "page": "Bridge.jl",
    "title": "Bridge.soft",
    "category": "Function",
    "text": "soft(t, T1, T2)\n\nTime change mapping s in [T1, T2](U-time) tot`in[T1, T2](X`-time).\n\n\n\n"
},

{
    "location": "index.html#Bridge.nu",
    "page": "Bridge.jl",
    "title": "Bridge.nu",
    "category": "Function",
    "text": " nu(k,P)\n\n(Bin-wise) integral of the Levy measure nu(B_k).\n\n\n\n"
},

{
    "location": "index.html#Bridge.bridge",
    "page": "Bridge.jl",
    "title": "Bridge.bridge",
    "category": "Function",
    "text": "bridge(W, P, scheme! = euler!) -> Y\n\nIntegrate with scheme! and set Y[end] = P.v1.\n\n\n\n"
},

{
    "location": "index.html#Bridge.Vs",
    "page": "Bridge.jl",
    "title": "Bridge.Vs",
    "category": "Function",
    "text": "Vs(s, T1, T2, v, P)\n\nTime changed V for generation of U.\n\n\n\n"
},

{
    "location": "index.html#Bridge.logpdfnormal",
    "page": "Bridge.jl",
    "title": "Bridge.logpdfnormal",
    "category": "Function",
    "text": "logpdfnormal(x, A)\n\nlogpdf of centered gaussian with covariance A\n\n\n\n"
},

{
    "location": "index.html#Bridge.mdb!",
    "page": "Bridge.jl",
    "title": "Bridge.mdb!",
    "category": "Function",
    "text": "mdb(u, W, P)\nmdb!(copy(W), u, W, P)\n\nEuler scheme with the diffusion coefficient correction of the modified diffusion bridge.\n\n\n\n\n\n"
},

{
    "location": "index.html#Bridge.euler!",
    "page": "Bridge.jl",
    "title": "Bridge.euler!",
    "category": "Function",
    "text": "euler!(Y, u, W, P) -> X\n\nSolve stochastic differential equation dX_t = b(tX_t)dt + (tX_t)dW_t  using the Euler scheme in place.\n\n\n\n"
},

{
    "location": "index.html#Bridge.GammaProcess",
    "page": "Bridge.jl",
    "title": "Bridge.GammaProcess",
    "category": "Type",
    "text": "GammaProcess\n\nA GammaProcess with jump rate γ and inverse jump size λ has increments Gamma(t*γ, 1/λ) and Levy measure\n\n(x)= x^-1exp(- x) \n\nHere Gamma(α,θ) is the Gamma distribution in julia's parametrization with shape parameter α and scale θ.\n\n\n\n"
},

{
    "location": "index.html#Bridge.dotVs",
    "page": "Bridge.jl",
    "title": "Bridge.dotVs",
    "category": "Function",
    "text": "dotVs (s, T1, T2, v, P)\n\nTime changed time derivative of V for generation of U.\n\n\n\n"
},

{
    "location": "index.html#Bridge.mdb",
    "page": "Bridge.jl",
    "title": "Bridge.mdb",
    "category": "Function",
    "text": "mdb(u, W, P)\nmdb!(copy(W), u, W, P)\n\nEuler scheme with the diffusion coefficient correction of the modified diffusion bridge.\n\n\n\n"
},

{
    "location": "index.html#Bridge.euler",
    "page": "Bridge.jl",
    "title": "Bridge.euler",
    "category": "Function",
    "text": "euler(u, W, P) -> X\n\nSolve stochastic differential equation dX_t = b(tX_t)dt + (tX_t)dW_t using the Euler scheme.\n\n\n\n"
},

{
    "location": "index.html#Bridge.mat",
    "page": "Bridge.jl",
    "title": "Bridge.mat",
    "category": "Function",
    "text": "mat(X::SamplePath{SVector}) \nmat(yy::Vector{SVector})\n\nReinterpret X or yy to an array without change in memory.\n\n\n\n"
},

{
    "location": "index.html#Bridge.thetamethod",
    "page": "Bridge.jl",
    "title": "Bridge.thetamethod",
    "category": "Function",
    "text": "thetamethod(u, W, P, theta=0.5)\n\nSolve stochastic differential equation using the theta method and Newton-Raphson steps.\n\n\n\n"
},

{
    "location": "index.html#Bridge.mcband",
    "page": "Bridge.jl",
    "title": "Bridge.mcband",
    "category": "Function",
    "text": "mcband(mc)\n\nCompute marginal 95% coverage interval for the chain from normal approximation.\n\n\n\n"
},

{
    "location": "index.html#Bridge.LinPro",
    "page": "Bridge.jl",
    "title": "Bridge.LinPro",
    "category": "Type",
    "text": "LinPro(B, μ::T, σ)\n\nLinear diffusion dX = B(X - )dt + dW\n\n\n\n"
},

{
    "location": "index.html#Unsorted-1",
    "page": "Bridge.jl",
    "title": "Unsorted",
    "category": "section",
    "text": "Bridge.ODESolver\nmcbandmean \nBridge.cumsum0 \nLocalGammaProcess\nBridge.Ptilde\nGuidedProp\nBridge.compensator0 \nmcstart \nCSpline \nBridge.tofs\nBridge.r\nBridge.θ \nBridge.gpK! \nmcnext \nBridge.compensator \nBridge.outer \nBridge.integrate \nBridge.soft\nBridge.nu \nbridge\nBridge.Vs\nBridge.logpdfnormal\nBridge.mdb!\neuler! \nGammaProcess\nBridge.dotVs\nBridge.mdb \neuler \nBridge.mat \nthetamethod \nmcband \nLinPro"
},

]}
