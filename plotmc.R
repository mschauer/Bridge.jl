library(ggplot2); library(gridExtra);library(plyr);library(GGally)
psep = '/'

NUM=1
thin = 1
burnin = 500
#if (Sys.info()["sysname"] == "Darwin") setwd("~/...")

simid <- 6
simname <- c('exci','nonexci','fullmbb', 'fullguip', 'atanguip100', 'atantcguip100', 'atanguip20', 'atantcguip20')[simid]

thetas_all <- read.csv(paste("output", psep, simname, psep, "params",".txt",sep=""),header=TRUE, sep=" ")
trueparam = (read.csv(paste("output", psep, simname, psep, "truth",".txt",sep=""),header=TRUE, sep=" "))
#trueparam$beta <- NULL
#thetas_all$beta <- NULL

params <- labels(trueparam)[[2]]
paramsfn <- params

colnames(thetas_all) <- c('n',params)
thetas_all['n'] <- NULL
N <- nrow(thetas_all)
NP <- length(params)

cs = 1:NP
prs= 1:NP
acf.col = 3

d_all <- stack(thetas_all)
thetas_all$m <- rep(simname,each=N)
head(d_all)
head(thetas_all)

 
d_all['m'] <- rep(rep(simname,each=N),NP)

names(d_all) <- c('value', 'parameter', 'm')


d_all$iterate <- rep(1:N,NUM*NP)
d_all$yintercept <- rep(c(as.matrix(trueparam)), each=NUM*N)

textlarge <- theme(text = element_text(size=18))

iterates_first_fig <- ggplot(subset(d_all,iterate<=burnin), aes(x=iterate,y=value)) + geom_path() + facet_grid(parameter~m,scales='free',labeller= label_parsed)  + geom_hline(aes(yintercept=yintercept),color='red') + ylab("") + textlarge +theme(axis.text.x=element_text(angle=90,hjust=1,vjust=.5))


d_all2 <- subset(d_all,iterate>burnin) # burnin removed
d_all2thin <- subset(d_all2,(iterate%%thin)==0)
iterates_thin_fig <- ggplot(d_all2thin, aes(x=iterate,y=value)) + geom_path() + facet_grid(parameter~m,scales='free',labeller= label_parsed)  + geom_hline(aes(yintercept=yintercept),color='red') + ylab("") + textlarge+ textlarge +theme(axis.text.x=element_text(angle=90,hjust=1,vjust=.5))

den_fig = list(NP)
# density plots
i = 1
for (i in 1:NP)
{
den_fig[[i]] <- ggplot(subset(d_all2,parameter==params[i]), aes(x=value,colour=m,linetype=m)) + geom_line(stat='density',size=1.5) + facet_grid(.~parameter,scales='free',labeller= label_parsed)   +xlab("") + textlarge + scale_colour_discrete(guide=FALSE) +scale_linetype_discrete(guide=FALSE) + facet_grid(.~parameter,scales='free_x',labeller= label_parsed)    
}


#### acf plots
lagmax <- 18

ind <- sort(unique(d_all2thin$iterate))
acfs = list(NP)
# density plots

for (i in 1:length(cs))
{
    acfs[[i]] <-acf(thetas_all[ind,cs[i]],lag.max = lagmax,plot=F)$acf
}
if (NP == 6) acfs2 <- c(c(acfs[[1]], acfs[[2]]),c(acfs[[3]], acfs[[4]]), c(acfs[[5]], acfs[[6]]))
if (NP == 3) acfs2 <- c(c(acfs[[1]], acfs[[2]], acfs[[3]]))
    
acf.df <- data.frame(lag=rep(0:lagmax,length(cs)),
                     acf=acfs2,
                     parameter=rep(params[cs],each=NUM*(lagmax+1  )))
head(acf.df)
acf_fig <- ggplot(data=acf.df, aes(x=lag, y=acf)) + 
  geom_hline(aes(yintercept = 0)) +
  geom_segment(mapping = aes(xend = lag, yend = 0))+ facet_wrap(~parameter, ncol=acf.col)+ textlarge



pdf(paste("output", psep, simname,psep,'pairs.pdf',sep=''))
par(mfrow=c(1,NUM))
#pairs plots


tt <- subset(thetas_all,m==simname)
colnames(tt) <- paramsfn
show(ggpairs(tt[seq(100, 100000,10),prs],title='',diag=list(continuous='density', combo = "box"), axisLabels="show",params=list(cex=0.8)))
 

dev.off();


ddply(d_all2, parameter~m, summarise, 
      mean = round(mean(value),4), 
      sd = round(sd(value),4) )


pdf(paste("output", psep, simname,psep,'iteratesfirst.pdf',sep=''))
show(iterates_first_fig)
dev.off()

pdf(paste("output", psep, simname,psep,'iteratesthin.pdf',sep=''))
show(iterates_thin_fig)
dev.off()

for (i in 1:NP)
{
  pdf(paste("output", psep, simname,psep,'den', paramsfn[i], '.pdf',sep=''))
  show(den_fig[[i]])
  dev.off()
}

 

pdf(paste("output", psep, simname,psep,'acf.pdf',sep=''))
show(acf_fig)
dev.off()

#source("plotmc.R")
