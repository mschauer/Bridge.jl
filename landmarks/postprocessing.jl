## postprocessing

fn = "_" * string(model) * "_" * string(sampler) *"_" * string(dataset)
gif(anim, outdir*fn*".gif", fps = 100)
mp4(anim, outdir*fn*".mp4", fps = 100)

######### write mcmc iterates of bridges to csv file
iterates = reshape(vcat(Xsave...),2*d*length(tt_)*P.n, length(subsamples)) # each column contains samplepath of an iteration
# Ordering in each column is as follows:
# 1) time
# 2) landmark nr
# 3) for each landmark: q1, q2 p1, p2
pqtype = repeat(["pos1", "pos2", "mom1", "mom2"], length(tt_)*P.n)
times = repeat(tt_,inner=2d*P.n)
landmarkid = repeat(1:P.n, inner=2d, outer=length(tt_))

out = hcat(times,pqtype,landmarkid,iterates)
head = "time " * "pqtype " * "landmarkid " * prod(map(x -> "iter"*string(x)*" ",subsamples))
head = chop(head,tail=1) * "\n"

fn = outdir*"iterates.csv"
f = open(fn,"w")
write(f, head)
writedlm(f,out)
close(f)

########### write info to txt file
fn = outdir*  "info_" * string(model) * "_" * string(sampler) *"_" * string(dataset)*".txt"
f = open(fn,"w")
write(f, "Dataset: ", string(dataset),"\n")
write(f, "Sampler: ", string(sampler), "\n")

write(f, "Number of iterations: ",string(ITER),"\n")
write(f, "Number of landmarks: ",string(P.n),"\n")
write(f, "Length time grid: ", string(length(tt_)),"\n")
write(f, "Mesh width: ",string(dt),"\n")
write(f, "Noise Sigma: ",string(σobs),"\n")
write(f, "rho (Crank-Nicholsen parameter: ",string(ρ),"\n")
write(f, "MALA parameter (delta): ",string(δ),"\n")
write(f, "skip in evaluation of loglikelihood: ",string(sk),"\n")
write(f, "Average acceptance percentage (path - initial state): ",string(perc_acc),"\n\n")
#write(f, "Backward type parametrisation in terms of nu and H? ",string(Î½Hparam),"\n")
close(f)

######## write observations to file
obsdf = DataFrame(x=vcat( extractcomp(xobs0,1), extractcomp(xobsT,1)),
                y= vcat( extractcomp(xobs0,2), extractcomp(xobsT,2)),
                time=repeat(["0","T"], inner=P.n))
CSV.write(outdir*"observations.csv", obsdf; delim=";")

######## write parameter iterates to file
parsdf = DataFrame(a=extractcomp(parsave,1),c=extractcomp(parsave,2),
            gamma=extractcomp(parsave,3), iterate=subsamples)
CSV.write(outdir*"parameters.csv", parsdf; delim=";")

####### write noisefields to file
if isa(P,Landmarks)
    nfsloc = [P.nfs[j].δ for j in eachindex(P.nfs)]
    nfsdf = DataFrame(locx =  extractcomp(nfsloc,1),
                      locy =  extractcomp(nfsloc,2),
                      nfstd=fill(P.nfstd,length(P.nfs)))
elseif isa(P,MarslandShardlow)
    nfsdf =DataFrame(locx=Int64[], locy=Int64[], nfstd=Int64[])
end
CSV.write(outdir*"noisefields.csv", nfsdf; delim=";")
