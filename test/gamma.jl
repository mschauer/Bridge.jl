using Base.Test, Bridge

# test the tricky bridge sampling when endpoint is or is not in grid
srand(10)
G =  GammaProcess(10.0, 1.5)
GB = GammaBridge(1.0, 2.0, G)
n = 1000
@test abs(mean(sample([0.0, 1.0, 3.0],  G).yy[end] for i in 1:1000) -  mean(Bridge.increment(3, G))) < 3*std(Bridge.increment(0.5, G))/sqrt(n)

X1 = sample([0.0, 0.5, 1.0], GB, 0.2)
X2 = sample([0.0, 0.5, 2.0], GB, 0.2)
X3 = sample([0.0, 0.5], GB, 0.2)

@test  X1.tt ≈ [0,1,2]/2
@test  X2.tt ≈ [0,1,4]/2
@test  X3.tt ≈ [0,1]/2

@test  X1.yy[1] == X1.yy[1] == X1.yy[1] == 0.2

@test  X1.yy[3] ≈ 2.0
@test  X2.yy[3] >= 2.0
@test  X3.yy[2] <= 2.0
