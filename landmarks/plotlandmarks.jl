@rlibrary ggforce

"""
    Plot landmark positions of guided of forward simulated path
    X: sample path containing landmark evolution
    Xᵒ: guided sample path containing landmark evolution
    n: number of landmarks
    model: either :ms or :ahs
    v0: starting configuration (positions)
    vT: ending configuration (positions)
    nfs: contains info over noisefields
    db: domainbound (plot on [-db,db] x [-db,db])
"""
function plotlandmarkpositions(X,Xᵒ,P,v0,vT;db=5)
    n = P.n
    if isa(P,Landmarks)
        nfs = P.nfs
    elseif isa(P,MarslandShardlow)
        nfs = 0.0
    end
    # construct df for X
    out = [Any[X.tt[i], [X.yy[i][CartesianIndex(c, k)][l] for l in 1:d, c in 1:2]..., "point$k"] for k in 1:n, i in eachindex(X.tt) ][:]
    df = DataFrame(time=extractcomp(out,1),pos1=extractcomp(out,2),pos2=extractcomp(out,3),mom1=extractcomp(out,4),mom2=extractcomp(out,5),pointID=extractcomp(out,6))
    # construct df for Xᵒ
    outg = [Any[Xᵒ.tt[i], [Xᵒ.yy[i][CartesianIndex(c, k)][l] for l in 1:d, c in 1:2]..., "point$k"] for k in 1:n, i in eachindex(Xᵒ.tt) ][:]
    dfg = DataFrame(time=extractcomp(outg,1),pos1=extractcomp(outg,2),pos2=extractcomp(outg,3),mom1=extractcomp(outg,4),mom2=extractcomp(outg,5),pointID=extractcomp(outg,6))
    # construct df for noisefields
    if isa(P,Landmarks)
        nfscales = [nfs[j].γ for j in eachindex(nfs)]
        nfsloc = [nfs[j].δ for j in eachindex(nfs)]
        df_nfs = DataFrame(locx =  extractcomp(nfsloc,1), locy =  extractcomp(nfsloc,2),
        lambda1=  extractcomp(nfscales,1), lambda2=extractcomp(nfscales,2), nfstd=fill(nfstd,length(nfs)))
    elseif isa(P,MarslandShardlow)
        df_nfs =DataFrame(locx=Int64[], locy=Int64[], lambda1=Int64[], lambda2=Int64[],nfstd=Int64[])
    end

    # construct df for initial and final configs
    # (need to repeat to connect also beginning and end)
    dfT = DataFrame(pos1=repeat(extractcomp(vT,1),2), pos2=repeat(extractcomp(vT,2),2))
    df0= DataFrame(pos1=repeat(extractcomp(v0,1),2), pos2=repeat(extractcomp(v0,2),2))

    if isa(P,MarslandShardlow)
        titel = "Marsland-Shardlow model, "
    elseif isa(P,Landmarks)
        titel = "Arnaudon-Holm-Sommer model, "
    end
    titel = titel * string(n)*" landmarks"

    @rput titel
    @rput df
    @rput dfg
    @rput df_nfs
    @rput dfT
    @rput df0
    @rput db

    R"""
    library(tidyverse)
    library(ggforce) # for circles
    dfpaths <- rbind(df,dfg)
    dfpaths$fg <- rep(c("forward","guided"),each=nrow(df))

    g <- ggplot() +
    geom_segment(data=df_nfs,aes(x=locx, y=locy, xend=locx, yend=locy+lambda2), arrow = arrow(length = unit(.1, "cm")),color="Grey")+
    geom_segment(data=df_nfs,aes(x=locx, y=locy, xend=locx+lambda1, yend=locy), arrow = arrow(length = unit(.1, "cm")),color="Grey")+
     geom_circle(aes(x0 = locx, y0 = locy, r = nfstd), data = df_nfs,color="Grey",linetype="dashed") +
     geom_path(data=df0, mapping=aes(x=pos1,y=pos2),colour='black',size=0.7) +
     geom_path(data=dfT, mapping=aes(x=pos1,y=pos2),colour='orange',size=0.7) +
     geom_point(data=dfT, mapping=aes(x=pos1,y=pos2),colour='orange',size=0.7) +
     geom_point(data=df0, mapping=aes(x=pos1,y=pos2),colour='black',size=0.7)+
     geom_path(data=dfpaths, mapping=aes(x=pos1,pos2,group=pointID,colour=time)) +
          facet_wrap(~fg) + geom_point(data=df_nfs, aes(x=locx,y=locy),size=2,color="Grey")+
              theme_light() + xlab('horizontal position') +
               coord_cartesian(xlim = c(-db,db), ylim = c(-db,db)) +
          ylab('vertical position') + ggtitle(titel)
    show(g)
    """
end

"""
    Plot landmark positions of both guided and forward simulated paths
    X: sample path containing landmark evolution
    n: number of landmarks
    model: either :ms or :ahs
    v0: starting configuration (positions)
    vT: ending configuration (positions)
    nfs: contains info over noisefields
    db: domainbound (plot on [-db,db] x [-db,db])
"""
function plotlandmarkpositions(X,P,v0,vT;db=5)
    n = P.n
    if isa(P,Landmarks)
        nfs = P.nfs
    elseif isa(P,MarslandShardlow)
        nfs = 0.0
    end
    # construct df for X
    out = [Any[X.tt[i], [X.yy[i][CartesianIndex(c, k)][l] for l in 1:d, c in 1:2]..., "point$k"] for k in 1:n, i in eachindex(X.tt) ][:]
    df = DataFrame(time=extractcomp(out,1),pos1=extractcomp(out,2),pos2=extractcomp(out,3),mom1=extractcomp(out,4),mom2=extractcomp(out,5),pointID=extractcomp(out,6))
    # construct df for noisefields
    if isa(P,Landmarks)
        nfscales = [nfs[j].γ for j in eachindex(nfs)]
        nflocs = [nfs[j].δ for j in eachindex(nfs)]
        df_nfs = DataFrame(locx =  extractcomp(nflocs,1), locy =  extractcomp(nflocs,2),
        lambda1=  extractcomp(nfscales,1), lambda2=extractcomp(nfscales,2), nfstd=fill(nfstd,length(nfs)))
    elseif isa(P,MarslandShardlow)
        df_nfs =DataFrame(locx=Int64[], locy=Int64[], lambda1=Int64[], lambda2=Int64[],nfstd=Int64[])
    end

    # construct df for initial and final configs
    # (need to repeat to connect also beginning and end)
    dfT = DataFrame(pos1=repeat(extractcomp(vT,1),2), pos2=repeat(extractcomp(vT,2),2))
    df0= DataFrame(pos1=repeat(extractcomp(v0,1),2), pos2=repeat(extractcomp(v0,2),2))

    if isa(P,MarslandShardlow)
        titel = "Marsland-Shardlow model, "
    elseif isa(P,Landmarks)
        titel = "Arnaudon-Holm-Sommer model, "
    end
    titel = titel * string(n)*" landmarks"

    @rput titel
    @rput df

    @rput df_nfs
    @rput dfT
    @rput df0
    @rput db

    R"""
    library(tidyverse)
    library(ggforce) # for circles
    dfpaths <- df

    g <- ggplot() +
    geom_segment(data=df_nfs,aes(x=locx, y=locy, xend=locx, yend=locy+lambda2), arrow = arrow(length = unit(.1, "cm")),color="Grey")+
    geom_segment(data=df_nfs,aes(x=locx, y=locy, xend=locx+lambda1, yend=locy), arrow = arrow(length = unit(.1, "cm")),color="Grey")+
     geom_circle(aes(x0 = locx, y0 = locy, r = nfstd), data = df_nfs,color="Grey",linetype="dashed") +
     geom_path(data=df0, mapping=aes(x=pos1,y=pos2),colour='black',size=0.7) +
     geom_path(data=dfT, mapping=aes(x=pos1,y=pos2),colour='orange',size=0.7) +
     geom_point(data=dfT, mapping=aes(x=pos1,y=pos2),colour='orange',size=0.7) +
     geom_point(data=df0, mapping=aes(x=pos1,y=pos2),colour='black',size=0.7)+
     geom_path(data=dfpaths, mapping=aes(x=pos1,pos2,group=pointID,colour=time)) +
              theme_light() + xlab('horizontal position') +
              coord_cartesian(xlim = c(-db,db), ylim = c(-db,db))+
          ylab('vertical position') + ggtitle(titel)
    show(g)
    """
end



"""
    On the square grid provided by (-db:δ:db) x (-db:δ:db) compute
    σq(q) and make heatmap
"""
function plotσq(db,nfs ;δ=0.1)
    rfine = -db:δ:db
    nfloc_fineres = Point.(collect(product(rfine, rfine)))[:]
    σq_fineres = σq(nfs).(nfloc_fineres)
    df_sigmaq = DataFrame(x=extractcomp(nfloc_fineres,1), y=extractcomp(nfloc_fineres,2), horiz =map(x->x[1,1], σq_fineres), vertic = map(x->x[2,2], σq_fineres) )
    if model ==:ahs
        nfscales = [nfs[j].γ for j in eachindex(nfs)]
        nfsloc = [nfs[j].δ for j in eachindex(nfs)]
        df_nfs = DataFrame(locx =  extractcomp(nfsloc,1), locy =  extractcomp(nfsloc,2),
        lambda1=  extractcomp(nfscales,1), lambda2=extractcomp(nfscales,2), nfstd=fill(nfstd,length(nfs)))
    else
        df_nfs =DataFrame(locx=Int64[], locy=Int64[], lambda1=Int64[], lambda2=Int64[],nfstd=Int64[])
    end
    @rput df_sigmaq
    @rput df_nfs
    R"""
    library(tidyverse)
    library(gridExtra)
    phor <- ggplot() + geom_point(data=df_sigmaq, aes(x=x, y=y,colour=horiz)) +
        ggtitle("Horizontal noise field on position")#+ geom_point(data=df_nfs, aes(x=locx,y=locy),size=2,color="Grey")
    pvert <- ggplot() + geom_point(data=df_sigmaq, aes(x=x, y=y,colour=vertic)) +
        ggtitle("Vertical noise field on position")#+ geom_point(data=df_nfs, aes(x=locx,y=locy),size=2,color="Grey")
    grid.arrange(phor,pvert)
    """
end

# R"""
# dfg %>% dplyr::select(time,mom1,mom2,pointID) %>%
#     filter(pointID %in% c("point1","point2","point3","point40","point41","point42")) %>%
#      ggplot(aes(x=mom1,mom2,group=pointID,colour=time)) + geom_path() +
#      facet_wrap(~pointID,scales="free") + theme_light()
# """


if false
    R"""
    dfg %>% dplyr::select(time,mom1,mom2,pointID) %>%
    filter(pointID %in% c("point1","point2","point3","point40","point41","point42")) %>%
     ggplot(aes(x=mom1,mom2,group=pointID,colour=time)) + geom_path() +
     facet_wrap(~pointID) + theme_light()
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
          theme_light() + xlab('horizontal position') +
          ylab('vertical position') + ggtitle(titel)
    """
end
if false
    R"""
    df %>% dplyr::select(time,mom1,mom2,pointID) %>%
        filter(pointID %in% c("point1","point2","point3","point40","point41","point42")) %>%
         ggplot(aes(x=mom1,mom2,group=pointID,colour=time)) + geom_path() +
         facet_wrap(~pointID,scales="free") + theme_light()
    """
end
