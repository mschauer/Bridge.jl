setwd("~/.julia/dev/Bridge/landmarks/figs")
library(tidyverse)
library(stringr)
library(gridExtra)
library(ggforce)
library(GGally)
theme_set(theme_light())
########  read observations

obsdf <- read_delim("observations.csv", ";", escape_double = FALSE, trim_ws = TRUE) 
n <- nrow(obsdf)/2
v0 <- obsdf[1:n,]; v0 <- rbind(v0,v0[1,])
vT = obsdf[seq(n+1,2*n),]; vT <- rbind(vT,vT[1,])

#######  read noisefields
nfsdf <- read_delim("noisefields.csv",";", escape_double = FALSE, trim_ws = TRUE)

######## read and analyse bridge iterates
d <- read_table2("iterates.csv") %>% mutate(landmarkid=as.factor(landmarkid))
d <- d %>% gather(key="iteratenr",value="value",contains("iter"))
d <- d %>% spread(key=pqtype, value=value)  %>% mutate(iteratenr=str_replace(iteratenr,"iter",""))
d <- d %>% mutate(iterate=as.numeric(str_replace(iteratenr,"iter","")))

####### read parameter updates
parsdf <- read_delim("parameters.csv", ";", escape_double = FALSE, trim_ws = TRUE) 

#------------------ figures --------------------------------------------------------------

shapes <- ggplot() +
  geom_path(data=v0, aes(x=x,y=y), colour='black')+
  geom_path(data=vT, aes(x=x,y=y), colour='orange') +
  geom_point(data=noisefields, aes(x=locx, y=locy), color="Grey")+
  geom_circle(aes(x0 = locx, y0 = locy, r = nfstd), data = noisefields,color="Grey",linetype="dashed")+ 
    theme(axis.title.x=element_blank(), axis.title.y=element_blank())

pdf("shapes_noisefields.pdf",width=7,height=4)  
show(shapes)
dev.off()

# plots paths of landmarks positions and bridges over various iterations
dsub <- d %>% dplyr::filter(iterate %in% c(0,5,100,200,300,400)) %>%
                        mutate(iteratenr = fct_relevel(iteratenr, c("0","5","100","200")))

p4 <- dsub %>% ggplot(aes(x=pos1,y=pos2)) + 
  geom_path(aes(group=interaction(landmarkid,iteratenr),colour=time)) + facet_wrap(~iteratenr) +
  geom_point(data=v0, aes(x=x,y=y), colour='black')+geom_point(data=vT, aes(x=x,y=y), colour='orange')+
  geom_path(data=v0, aes(x=x,y=y), colour='black')+geom_path(data=vT, aes(x=x,y=y), colour='orange') +
  theme(axis.title.x=element_blank(), axis.title.y=element_blank())
pdf("iterats_faceted.pdf",width=7,height=4)  
show(p4)
dev.off()

# plot overlaid landmark bridges
p1 <- d %>% dplyr::filter(iterate %in% seq(0,400,by=50))  %>% ggplot() + 
  geom_path(aes(pos1,y=pos2,group=interaction(landmarkid,iteratenr),colour=iterate)) +
  scale_colour_gradient(low="green",high="darkblue")+ 
  geom_point(data=v0, aes(x=x,y=y), colour='black')+geom_point(data=vT, aes(x=x,y=y), colour='orange')+
  geom_path(data=v0, aes(x=x,y=y), colour='black',size=1.1)+geom_path(data=vT, aes(x=x,y=y), colour='orange',size=1.1) +
  theme(axis.title.x=element_blank(), axis.title.y=element_blank())
pdf("manypaths.pdf",width=7,height=4)  
show(p1)
dev.off()

# plot parameter updates
ppar1 <- parsdf %>% gather(key=par, value=value, a, c, gamma) %>% 
  ggplot(aes(x=iterate, y=value)) + geom_path() + facet_wrap(~par, scales="free_y") + xlab("iterate") + ylab("")
pdf("tracepars.pdf",width=7,height=3)  
  show(ppar1)
dev.off()


ppar2 <- parsdf %>% ggplot(aes(x=a,y=c,colour=iterate)) + geom_point() + theme(legend.position = 'none')
ppar3 <- parsdf %>% ggplot(aes(x=a,y=gamma,colour=iterate)) + geom_point() + theme(legend.position = 'none')
ppar4 <- parsdf %>% ggplot(aes(x=c,y=gamma,colour=iterate)) + geom_point()+ theme(legend.position = 'none')
pdf("scatterpars.pdf",width=7,height=3)  
grid.arrange(ppar2,ppar3,ppar4,ncol=3)
dev.off()




# plot paths of landmarks momenta
d %>% ggplot(aes(x=mom1,y=mom2)) + 
  geom_path(aes(group=interaction(landmarkid,iteratenr),colour=iteratenr_)) +
  facet_wrap(~landmarkid)  +scale_colour_gradient(low="orange",high="darkblue")

# some traceplots



