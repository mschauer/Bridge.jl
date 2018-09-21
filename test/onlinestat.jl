using Bridge, LinearAlgebra, Test, Statistics
x = [rand(5) for i in 1:10]    
m, sc, m2 = Bridge.iterateall(Bridge.MeanCov(x))    
@test norm(cov(x, corrected=true) - sc*m2) < eps(100.0)
@test norm(mean(x) - m) < eps(100.0)

y = [rand() for i in 1:10]    
m, sc, m2 = Bridge.iterateall(Bridge.MeanCov(y))    
@test norm(var(y, corrected=true) - sc*m2) < eps(100.0)
@test norm(mean(y) - m) < eps(100.0)

@inferred Bridge.iterateall(Bridge.MeanCov(1:1000))
@inferred Bridge.iterateall(Bridge.MeanCov(x))