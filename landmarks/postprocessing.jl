## postprocessing

fn = "_" * string(model) * "_" * string(sampler) *"_" * string(dataset)
gif(anim, outdir*fn*".gif", fps = 50)
mp4(anim, outdir*fn*".mp4", fps = 50)

# make a fig for acceptance probs in parameter and initial state updating
accdf = DataFrame(kernel = map(x->x.kernel, accinfo), acc = map(x->x.acc, accinfo), iter = 1:length(accinfo))
@rput accdf
@rput outdir
R"""
library(tidyverse)
library(ggplot2)
theme_set(theme_light())
p <-    accdf %>% mutate(kernel=as.character(kernel)) %>%
        ggplot(aes(x=iter, y=acc)) + geom_point() +
        facet_wrap(~kernel)
ggsave(paste0(outdir,"acceptance.pdf"),p)
"""


######### write mcmc iterates of bridges to csv file
nshapes = length(xobsT)
iterates = reshape(vcat(Xsave...),2*d*length(tt_)*P.n*nshapes, length(subsamples)) # each column contains samplepath of an iteration
# Ordering in each column is as follows:
# 0) shape
# 1) time
# 2) landmark nr
# 3) for each landmark: q1, q2 p1, p2
pqtype = repeat(["pos1", "pos2", "mom1", "mom2"], length(tt_)*P.n*nshapes)
times = repeat(tt_,inner=2d*P.n*nshapes)
landmarkid = repeat(1:P.n, inner=2d, outer=length(tt_)*nshapes)
shapes = repeat(1:nshapes, inner=length(tt_)*2d*P.n)

out = hcat(times,pqtype,landmarkid,shapes,iterates)
headline = "time " * "pqtype " * "landmarkid " * "shapes " * prod(map(x -> "iter"*string(x)*" ",subsamples))
headline = chop(headline,tail=1) * "\n"

fn = outdir*"iterates.csv"
f = open(fn,"w")
write(f, headline)
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
write(f, "Average acceptance percentage pCN update steps: ",string(perc_acc_pcn),"\n\n")
#write(f, "Backward type parametrisation in terms of nu and H? ",string(Î½Hparam),"\n")
close(f)

######## write observations to file
# if obs_atzero
#     obsdf = DataFrame(x=vcat( extractcomp(xobs0,1), extractcomp(xobsT,1)),
#                 y= vcat( extractcomp(xobs0,2), extractcomp(xobsT,2)),
#                 time=repeat(["0","T"], inner=P.n))
#     CSV.write(outdir*"observations.csv", obsdf; delim=";")
# else
    valueT = vcat(map(x->deepvec(x), xobsT)...) # merge all observations at time T in one vector
    posT = repeat(["pos1","pos2"], P.n*nshapes)
    shT = repeat(1:nshapes, inner=d*P.n)
    obsTdf = DataFrame(pos=posT,shape=shT, value=valueT,landmark=repeat(1:P.n,inner=d,outer=nshapes))

    q0 = map(x->vec(x),x0.q)
    p0 = map(x->vec(x),x0.p)
    obs0df = DataFrame(pos1=extractcomp(q0,1), pos2=extractcomp(q0,2), mom1=extractcomp(p0,1) , mom2=extractcomp(p0,2),landmark=1:P.n)

    CSV.write(outdir*"obs0.csv", obs0df; delim=";")
    CSV.write(outdir*"obsT.csv", obsTdf; delim=";")
# end


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
