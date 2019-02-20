extractcomp(v,i) = map(x->x[i], v)


out = [Any[X.tt[i], [X.yy[i][CartesianIndex(c, k)][l] for l in 1:d, c in 1:2]..., "point$k"] for k in 1:n, i in eachindex(X.tt) ][:]
df = DataFrame(time=extractcomp(out,1),pos1=extractcomp(out,2),pos2=extractcomp(out,3),mom1=extractcomp(out,4),mom2=extractcomp(out,5),pointID=extractcomp(out,6))
#Any["X.tt[$i]", ["X.yy[$i][CartesianIndex($c, $k)][$l]" for l in 1:d, k in 1:n, c in 1:2]...]

if model == :ms
    titel = "Marsland-Shardlow model, "
else
    titel = "Arnaudon-Holm-Sommer model, "
end
titel = titel * string(n)*" landmarks"

dfT = DataFrame(pos1=extractcomp(vT,1), pos2=extractcomp(vT,2))
df0= DataFrame(pos1=extractcomp(v0,1), pos2=extractcomp(v0,2))

@rput titel
@rput df
@rput dfT
@rput df0

if false
    R"""
    library(tidyverse)
    df %>% dplyr::select(time,pos1,pos2,pointID) %>%
         ggplot() +
          geom_path(data=df,aes(x=pos1,pos2,group=pointID,colour=time)) +
          geom_point(data=dfT, mapping=aes(x=pos1,y=pos2),colour='orange',size=0.7) +
          geom_point(data=df0, mapping=aes(x=pos1,y=pos2),colour='black',size=0.7)+
          theme_minimal() + xlab('horizontal position') +
          ylab('vertical position') + ggtitle(titel)
    """


    R"""
    df %>% dplyr::select(time,mom1,mom2,pointID) %>%
        filter(pointID %in% c("point1","point2","point3","point40","point41","point42")) %>%
         ggplot(aes(x=mom1,mom2,group=pointID,colour=time)) + geom_path() +
         facet_wrap(~pointID,scales="free") + theme_minimal()
    """
end

#### plotting
outg = [Any[XX.tt[i], [XX.yy[i][CartesianIndex(c, k)][l] for l in 1:d, c in 1:2]..., "point$k"] for k in 1:n, i in eachindex(XX.tt) ][:]
dfg = DataFrame(time=extractcomp(outg,1),pos1=extractcomp(outg,2),pos2=extractcomp(outg,3),mom1=extractcomp(outg,4),mom2=extractcomp(outg,5),pointID=extractcomp(outg,6))
#Any["X.tt[$i]", ["X.yy[$i][CartesianIndex($c, $k)][$l]" for l in 1:d, k in 1:n, c in 1:2]...]

dfT = DataFrame(pos1=extractcomp(vT,1), pos2=extractcomp(vT,2))
df0= DataFrame(pos1=extractcomp(v0,1), pos2=extractcomp(v0,2))


if model == :ms
    titel = "Marsland-Shardlow model, "
else
    titel = "Arnaudon-Holm-Sommer model, "
end
titel = titel * string(n)*" landmarks"

@rput titel
@rput dfg
@rput dfT
@rput df0
@rput T
R"""
library(tidyverse)
T_ <- T
dfsub <- df %>% dplyr::select(time,pos1,pos2,pointID) %>% filter(time<T_)
dfsubg <- dfg %>% dplyr::select(time,pos1,pos2,pointID) %>% filter(time<T_)

sub <- rbind(dfsub,dfsubg)
sub$fg <- rep(c("forward","guided"),each=nrow(dfsub))

g <- ggplot() +
      geom_path(data=sub, mapping=aes(x=pos1,pos2,group=pointID,colour=time)) +
      geom_point(data=dfT, mapping=aes(x=pos1,y=pos2),colour='orange',size=0.7) +
      geom_point(data=df0, mapping=aes(x=pos1,y=pos2),colour='black',size=0.7)+
      facet_wrap(~fg,scales="free") +
      theme_minimal() + xlab('horizontal position') +
      ylab('vertical position') + ggtitle(titel)
show(g)
"""


R"""
dfg %>% dplyr::select(time,mom1,mom2,pointID) %>%
    filter(pointID %in% c("point1","point2","point3","point40","point41","point42")) %>%
     ggplot(aes(x=mom1,mom2,group=pointID,colour=time)) + geom_path() +
     facet_wrap(~pointID,scales="free") + theme_minimal()
"""


if false
    R"""
    dfg %>% dplyr::select(time,mom1,mom2,pointID) %>%
    filter(pointID %in% c("point1","point2","point3","point40","point41","point42")) %>%
     ggplot(aes(x=mom1,mom2,group=pointID,colour=time)) + geom_path() +
     facet_wrap(~pointID) + theme_minimal()
    """
    R"""
    library(tidyverse)
    dfsub <- df %>% dplyr::select(time,pos1,pos2,pointID) %>% filter(time<0.99)
    dfsubg <- dfg %>% dplyr::select(time,pos1,pos2,pointID) %>% filter(time<0.99)

    sub <- rbind(dfsub,dfsubg)
    sub$fg <- rep(c("forward","guided"),each=nrow(dfsub))

    ggplot() +
          geom_path(data=sub, mapping=aes(x=pos1,pos2,group=pointID,colour=fg)) +
          geom_point(data=dfT, mapping=aes(x=pos1,y=pos2),colour='orange',size=0.7) +
          geom_point(data=df0, mapping=aes(x=pos1,y=pos2),colour='black',size=0.7)+
          #facet_wrap(~fg) +
          theme_minimal() + xlab('horizontal position') +
          ylab('vertical position') + ggtitle(titel)
    """
end
