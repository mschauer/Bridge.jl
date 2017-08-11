using Bridge, Distributions, StaticArrays
using BenchmarkTools

suite = BenchmarkGroup()
suite["solver"] = BenchmarkGroup(["SVector","SMatrix"])


SV = SVector{2,Float64}
SM = SArray{Tuple{2,2},Float64,2,4}
B = SM([-1 0.1; -0.2 -1])
mu = 0*SV([0.2, 0.3])
sigma = SM(2*[-0.212887  0.0687025;
  0.193157  0.388997 ])
a = sigma*sigma'
  
P = Bridge.LinPro(B, mu, sigma)

u = SV([1.0, 0.0])
v = SV([.5, 0.0])


t = 0.5
T = 2.0
n2 = 150
tt = linspace(t, T, n2)
K = SamplePath(tt, zeros(SM, length(tt)))
V = SamplePath(tt, zeros(SV, length(tt)))
Mu = SamplePath(tt, zeros(SV, length(tt)))
suite["solver"]["gpK!"] = @benchmarkable Bridge.gpK!(K, P)
suite["solver"]["gpV!"] = @benchmarkable Bridge.gpV!(V, v, P)
suite["solver"]["solve!"] = @benchmarkable Bridge.solve!(Bridge._F, Mu, u, P)
W = sample(tt, Wiener{SV}())
X = euler(u, W, P)
suite["solver"]["sample!(..., ::Wiener)"] = @benchmarkable sample!(W, Wiener{SV}())
suite["solver"]["euler!(..., ::Wiener)"] = @benchmarkable euler!(X, u, W, Wiener{SV}())
suite["solver"]["euler!(..., ::LinPro)"] = @benchmarkable euler!(X, u, W, P);

if isdefined(:NEW)
    tune!(suite);
    BenchmarkTools.save(joinpath("test","perf","params.jld"), "suite", BenchmarkTools.params(suite))
else
     BenchmarkTools.loadparams!(suite, BenchmarkTools.load(joinpath("test","perf","params.jld"), "suite"), :evals, :samples);
end

results = run(suite, verbose = true, seconds = 5) #  _ seconds per benchmark
BenchmarkTools.save(joinpath("test","perf","results$(Base.Dates.today()).jld"), "oldresults", results)

oldresults = BenchmarkTools.load(joinpath("test","perf","results.jld"), "oldresults")

judge(median(oldresults["solver"]), median(results["solver"]),time_tolerance = 0.01)
#  judge(memory(oldresults["solver"]), memory(results["solver"]))
