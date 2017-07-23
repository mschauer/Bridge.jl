using Base.Test, Bridge

# test the tricky bridge sampling when endpoint is or is not in grid

G =  GammaProcess(10., 1.)
GB =  GammaBridge(1., 2., G)
X1 = sample([0., 0.5, 1.], GB, 0.2)
X2 = sample([0., 0.5, 2.], GB, 0.2)
X3 = sample([0., 0.5], GB, 0.2)

@test  X1.tt ≈ [0,1,2]/2
@test  X2.tt ≈ [0,1,4]/2
@test  X3.tt ≈ [0,1]/2

@test  X1.yy[1] == X1.yy[1] == X1.yy[1] == 0.2

@test  X1.yy[3] ≈ 2.0
@test  X2.yy[3] >= 2.0
@test  X3.yy[2] <= 2.0
