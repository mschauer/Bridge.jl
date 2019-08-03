setwd("~/.julia/dev/Bridge/landmarks/figs")
library(tidyverse)
library(stringr)
library(gridExtra)
d <- read_table2("iterates.csv") %>% mutate(landmarkid=as.factor(landmarkid))
d <- d %>% gather(key="iteratenr",value="value",contains("iter"))
d <- d %>% spread(key=pqtype, value=value) 
d <- d %>% mutate(iteratenr_=as.numeric(str_replace(iteratenr,"iter","")))

p1 <- d %>% ggplot(aes(x=pos1,y=pos2)) + 
  geom_path(aes(group=interaction(landmarkid,iteratenr),colour=iteratenr_)) +
  scale_colour_gradient(low="orange",high="darkblue")+
  theme(legend.position = "none")

p2 <- d %>% ggplot(aes(x=pos1,y=pos2)) + 
   geom_path(aes(group=interaction(landmarkid,iteratenr),colour=landmarkid)) + 
theme(legend.position = "none")

grid.arrange(p1,p2,ncol=2)

d %>% ggplot(aes(x=mom1,y=mom2)) + 
  geom_path(aes(group=interaction(landmarkid,iteratenr),colour=iteratenr_)) +
  facet_wrap(~landmarkid)  +scale_colour_gradient(low="orange",high="darkblue")
