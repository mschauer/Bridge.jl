using Bridge, Distributions, StaticArrays
using BenchmarkTools

suite = BenchmarkGroup()


suite["solver"] = BenchmarkGroup(["Float64"])
suite["solverSA"] = BenchmarkGroup(["SVector", "SMatrix"])

if isdefined(:NEW) && NEW
    ;
else
    BenchmarkTools.loadparams!(suite, BenchmarkTools.load(joinpath("test","perf","params.jld"), "suite"), :evals, :samples);
end


B = -0.5
mu = 0.2
sigma = 0.8
a = sigma*sigma'
  
P = Bridge.LinPro(B, mu, sigma)

u = 0.5
v = 1.0

t = 0.5
T = 2.0
n2 = 150
tt = range(t, stop=T, length=n2)
K = SamplePath(tt, zeros(length(tt)))
V = SamplePath(tt, zeros(length(tt)))
Mu = SamplePath(tt, zeros(length(tt)))
suite["solver"]["gpK!"] = @benchmarkable Bridge.gpHinv!(K, P)
suite["solver"]["gpV!"] = @benchmarkable Bridge.gpV!(V, v, P)
suite["solver"]["solve!(::R3, ...)"] = @benchmarkable Bridge.solve!(Bridge.R3(), Bridge._F, Mu, u, P)
suite["solver"]["solve!(::BS3, ...)"] = @benchmarkable Bridge.solve!(Bridge.BS3(), Bridge._F, Mu, u, P)

W = sample(tt, Wiener())
X = euler(u, W, P)
suite["solver"]["sample!(..., ::Wiener)"] = @benchmarkable sample!(W, Wiener())
suite["solver"]["euler!(..., ::Wiener"] = @benchmarkable euler!(X, u, W, Wiener())
suite["solver"]["euler!(..., ::LinPro)"] = @benchmarkable euler!(X, u, W, P);

if isdefined(:NEW) && NEW
    tune!(suite["solver"]);
    BenchmarkTools.save(joinpath("test","perf","params.jld"), "suite", BenchmarkTools.params(suite))
else
    BenchmarkTools.loadparams!(suite, BenchmarkTools.load(joinpath("test","perf","params.jld"), "suite"), :evals, :samples);
end

results = run(suite["solver"], verbose = true, seconds = 5) #  _ seconds per benchmark


##############

SV = SVector{2,Float64}
SM = SArray{Tuple{2,2},Float64,2,4}
B = SM([-1 0.1; -0.2 -1])
mu = SV([0.2, 0.3])
sigma = SM(2*[-0.212887  0.0687025;
  0.193157  0.388997 ])
a = sigma*sigma'
  
P = Bridge.LinPro(B, mu, sigma)

u = SV([1.0, 0.0])
v = SV([.5, 0.0])


t = 0.5
T = 2.0
n2 = 150
tt = range(t, stop=T, length=n2)
K = SamplePath(tt, zeros(SM, length(tt)))
V = SamplePath(tt, zeros(SV, length(tt)))
Mu = SamplePath(tt, zeros(SV, length(tt)))
suite["solverSA"]["gpK!"] = @benchmarkable Bridge.gpHinv!(K, P)
suite["solverSA"]["gpV!"] = @benchmarkable Bridge.gpV!(V, v, P)
suite["solverSA"]["solve!(::R3, ...)"] = @benchmarkable Bridge.solve!(Bridge.R3(), Bridge._F, Mu, u, P)
suite["solverSA"]["solve!(::BS3, ...)"] = @benchmarkable Bridge.solve!(Bridge.BS3(), Bridge._F, Mu, u, P)

W = sample(tt, Wiener{SV}())
X = euler(u, W, P)
suite["solverSA"]["sample!(..., ::Wiener)"] = @benchmarkable sample!(W, Wiener{SV}())
suite["solverSA"]["euler!(..., ::Wiener"] = @benchmarkable euler!(X, u, W, Wiener{SV}())
suite["solverSA"]["euler!(..., ::LinPro)"] = @benchmarkable euler!(X, u, W, P);

if isdefined(:NEW) && NEW
    tune!(suite["solverSA"]);
end

resultsSA = run(suite["solverSA"], verbose = true, seconds = 5) #  _ seconds per benchmark



if isdefined(:NEW) && NEW
    BenchmarkTools.save(joinpath("test","perf","params.jld"), "suite", BenchmarkTools.params(suite))
end


BenchmarkTools.save(joinpath("test","perf","results$(Base.Dates.today()).jld"), "oldresults", results, "oldresultsSA", resultsSA)

oldresults, oldresultsSA = BenchmarkTools.load(joinpath("test","perf","results.jld"), "oldresults", "oldresultsSA")
display(median(results))
display(median(resultsSA))
display(judge(median(results), median(oldresults),time_tolerance = 0.01))
display(judge(median(resultsSA), median(oldresultsSA),time_tolerance = 0.01))

#  judge(memory(oldresults["solver"]), memory(results["solver"]))
