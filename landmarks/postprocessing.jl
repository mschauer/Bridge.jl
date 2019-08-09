## postprocessing


# write mcmc iterates to csv file
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

println("Average acceptance percentage: ",perc_acc,"\n")
println("Elapsed time: ",round(elapsed;digits=3))



# write info to txt file
fn = outdir*"info.txt"
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



pardf = DataFrame(a=extractcomp(parsave,1),gamma=extractcomp(parsave,2), subsamples=subsamples)
@rput pardf
R"""
library(ggplot2)
pardf %>% ggplot(aes(x=a,y=gamma,colour=subsamples)) + geom_point()
"""

pp1 = Plots.plot(subsamples, extractcomp(parsave,1),label="a")
xlabel!(pp1,"iteration")
pp2 = Plots.plot(subsamples, extractcomp(parsave,2),label="γ")
xlabel!(pp2,"iteration")
pp3 = Plots.plot(extractcomp(parsave,1), extractcomp(parsave,2),seriestype=:scatter,label="")
xlabel!(pp3,"a")
ylabel!(pp3,"γ")
l = @layout [a  b c]
pp = Plots.plot(pp1,pp2,pp3,background_color = :ivory,layout=l , size = (900, 500) )

cd(outdir)
#Plots.savefig("trace_pars.pdf")
