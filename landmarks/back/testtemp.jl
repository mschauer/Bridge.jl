Qvec = [construct_guidedproposal!(tt_, guidrecvec[k], (LT,ΣT,μT), (L0, Σ0),
          (xobs0, xobsTvec[k]), P, Pauxvec[k]) for k in 1:nshapes]
Xvec = [initSamplePath(tt_, xinit) for i in 1:nshapes]
Wvec = [initSamplePath(tt_,  zeros(StateW, dwiener)) for i in 1:nshapes]
for i in 1:nshapes
      sample!(Wvec[i], Wiener{Vector{StateW}}())
end
ll = simguidedlm_llikelihood!(LeftRule(), Xvec, xinit, Wvec, Qvec; skip=sk)

Xvecᵒ = deepcopy(Xvec)
Xvecᵒ[1].yy[1] = 3*Xvec[1].yy[1]
Xvecᵒ[1].yy[1] - Xvec[1].yy[1]
