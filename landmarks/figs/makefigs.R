setwd("~/.julia/dev/Bridge/landmarks/figs")
library(tidyverse)
library(stringr)
library(gridExtra)
library(ggforce)
library(GGally)
theme_set(theme_light())
########  read observations
  
obs0df <- read_delim("obs0.csv", ";", escape_double = FALSE, trim_ws = TRUE)
obsTdf <- read_delim("obsT.csv", ";", escape_double = FALSE, trim_ws = TRUE) %>% spread(key=pos,value=value) %>%
    mutate(shape=as.factor(shape))


v0 <- bind_rows(obs0df, obs0df)
vT <- bind_rows(obsTdf, obsTdf)
vT1 <- vT %>% dplyr::filter(shape=="1")
  
#######  read noisefields
nfsdf <- read_delim("noisefields.csv",";", escape_double = FALSE, trim_ws = TRUE)
  
######## read and analyse bridge iterates
d <- read_table2("iterates.csv") %>%
        mutate(landmarkid=as.factor(landmarkid)) %>%
        gather(key="iteratenr",value="value",contains("iter")) 
d <- d %>% mutate(i = rep(1:(nrow(d)/4),each=4)) # add column to identify a particular combination (pos1,pos2,mom1,mom2) of one landmark at a specific time of a specific shape
d <- d %>% spread(key=pqtype, value=value) %>% dplyr::select(-i) %>%  # can drop identifier column
   mutate(iteratenr=str_replace(iteratenr,"iter",""),iterate=as.numeric(str_replace(iteratenr,"iter","")))
d

# select all data for shape 1
d1 <- d %>% dplyr::filter(shapes==1)
  
dsub <- d1 %>% dplyr::filter(iterate %in% c(0,5,10,25,600,700)) %>%
    mutate(iteratenr = fct_relevel(iteratenr, c("0","5","10","25")))

# v0 <- obsdf[1:n,] 
dlabel0 <- obs0df; dlabel0$landmarkid <- unique(d$landmarkid)
# v0 <- rbind(v0,v0[1,])
#   
# vT = obsdf[seq(n+1,2*n),]
dlabelT <- obsTdf; dlabelT$landmarkid <- unique(d$landmarkid)
# vT <- rbind(vT,vT[1,])
  
  
####### read parameter updates
parsdf <- read_delim("parameters.csv", ";", escape_double = FALSE, trim_ws = TRUE) 
  
  #------------------ figures --------------------------------------------------------------
# plots shapes and noisefields  
shapes <- ggplot() +
    geom_path(data=v0, aes(x=pos1,y=pos2), colour='black')+
    geom_path(data=vT, aes(x=pos1,y=pos2,group=shape), colour='orange') +
    geom_point(data=nfsdf, aes(x=locx, y=locy), color="Grey")+
    geom_circle(aes(x0 = locx, y0 = locy, r = nfstd), data = nfsdf,color="Grey",linetype="dashed")+ 
      theme(axis.title.x=element_blank(), axis.title.y=element_blank()) + 
    geom_label(data=dlabel0, aes(x=pos1,y=pos2,label=landmarkid))+
    geom_label(data=dlabelT, aes(x=pos1,y=pos2,label=landmarkid),colour="orange")
  
pdf("shapes-noisefields.pdf",width=6,height=4)  
  show(shapes)
dev.off()
  
# plot paths of landmarks positions and bridges over various iterations
p4 <-     dsub %>% ggplot(aes(x=pos1,y=pos2)) +
    geom_path(aes(group=interaction(landmarkid,iteratenr),colour=time)) + facet_wrap(~iteratenr) +
    geom_point(data=v0, aes(x=pos1,y=pos2), colour='black')+geom_point(data=vT, aes(x=pos1,y=pos2), colour='orange')+
    geom_path(data=v0, aes(x=pos1,y=pos2), colour='black')+geom_path(data=vT, aes(x=pos1,y=pos2,group=shape), colour='orange') +
    theme(axis.title.x=element_blank(), axis.title.y=element_blank()) + coord_fixed()
    
pdf("bridges-faceted.pdf",width=6,height=4)  
  show(p4)
dev.off()

# plot overlaid landmark bridges
p1 <- d1 %>% dplyr::filter(iterate %in% seq(0,max(d$iterate),by=5))  %>% ggplot() + 
    geom_path(aes(pos1,y=pos2,group=interaction(landmarkid,iteratenr),colour=iterate)) +
    scale_colour_gradient(low="orange",high="darkblue")+ 
    geom_point(data=v0, aes(x=pos1,y=pos2), colour='black')+geom_point(data=vT, aes(x=pos1,y=pos2), colour='orange')+
    geom_path(data=v0, aes(x=pos1,y=pos2), colour='black',size=1.1)+geom_path(data=vT, aes(x=pos1,y=pos2,group=shape), colour='orange',size=1.1) +
    theme(axis.title.x=element_blank(), axis.title.y=element_blank()) +
    geom_label(data=dlabel0, aes(x=pos1,y=pos2,label=landmarkid,hjust="outward",vjust="outward"))+
  geom_point(data=nfsdf, aes(x=locx, y=locy), color="Grey")+
  geom_circle(aes(x0 = locx, y0 = locy, r = nfstd), data = nfsdf,color="Grey",linetype="dashed")+ 
  coord_fixed()
pdf("bridges-overlaid.pdf",width=6,height=4)  
  show(p1)
dev.off()
  
# plot parameter updates
ppar1 <- parsdf %>% mutate(cdivgamma2=c/gamma^2) %>% gather(key=par, value=value, a, c, gamma,cdivgamma2) 
ppar1$par <- factor(ppar1$par, levels=c('a', 'c', 'gamma','cdivgamma2'), labels=c("a","c",expression(gamma),expression(c/gamma^2)))
tracepars <- ppar1 %>% ggplot(aes(x=iterate, y=value)) + geom_path() + facet_wrap(~par, scales="free_y",labeller = label_parsed) +
 xlab("iterate") + ylab("") +  theme(strip.text.x = element_text(size = 12))
pdf("trace-pars.pdf",width=6,height=4)  
  show(tracepars)
dev.off()
  
# pairwise scatter plots for parameter updates  
ppar2 <- parsdf %>% ggplot(aes(x=a,y=c,colour=iterate)) + geom_point() + theme(legend.position = 'none')  +scale_colour_gradient(low="orange",high="darkblue")
ppar3 <- parsdf %>% ggplot(aes(x=a,y=gamma,colour=iterate)) + geom_point() + theme(legend.position = 'none') +scale_colour_gradient(low="orange",high="darkblue")
ppar4 <- parsdf %>% ggplot(aes(x=c,y=gamma,colour=iterate)) + geom_point()+ theme(legend.position = 'none') +scale_colour_gradient(low="orange",high="darkblue")
pdf("scatter-pars.pdf",width=6,height=2)  
  grid.arrange(ppar2,ppar3,ppar4,ncol=3)
dev.off()
  
  
# plot paths of landmarks momenta
pmom <-  d %>% dplyr::filter(time==0) %>% ggplot(aes(x=mom1,y=mom2,colour=iterate)) + geom_point() +
  #  geom_path(aes(group=interaction(landmarkid,iteratenr),colour=iterate)) +
    facet_wrap(~landmarkid)  +scale_colour_gradient(low="orange",high="darkblue")+theme(axis.title.x=element_blank(), axis.title.y=element_blank()) +
  geom_hline(yintercept=0, linetype="dashed")+geom_vline(xintercept=0, linetype="dashed")
pdf("momenta-faceted.pdf",width=6,height=4)  
  show(pmom)
dev.off()


# plot initial positions for all shapes (only interesting in case initial shape is unobserved)
dtime0 <- d %>% dplyr::filter(time==0)# ,iterate>200) 

# add factor for determining which phase of sampling
dtime0 <-  dtime0 %>% mutate(phase = 
    case_when(iterate < quantile(iterate,1/3) ~ "initial",
              between(iterate,quantile(iterate,1/3),quantile(iterate,2/3)) ~ "middle",
              iterate >= quantile(iterate,1/3) ~ "end")  ) %>% # reorder factor levels
  mutate(phase = fct_relevel(phase, "initial", "middle"))
dtimeT <- d %>% dplyr::filter(time==1) 
initshapes0 <- ggplot()  + 
  geom_point(data=vT, aes(x=pos1,y=pos2), colour='grey')+
  geom_path(data=vT, aes(x=pos1,y=pos2,group=shape), colour='grey', linetype="dashed",size=0.4) +
    geom_point(data=v0, aes(x=pos1,y=pos2), colour='red')+
  geom_path(data=v0, aes(x=pos1,y=pos2), colour='red',size=0.8,alpha=0.8) +
    geom_path(data=bind_rows(dtime0,dtime0),aes(x=pos1,y=pos2,colour=iterate)) +
  facet_wrap(~phase)+ coord_fixed()
initshapes0



pdf("initial-shapes.pdf",width=6,height=2)  
initshapes0
dev.off()

