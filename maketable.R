library(coda)
library(mcmcse)
psep = "/"
methods <- c("exdhmdb","exaffeul","exafftcs","exafftheta")
methodabbr <- c("MDB", "GP", "GP-TC", "GP-TH", "GP-TC/TH")
stepsizes <- c(10, 25, 100)
params = c("\\vartheta_1", "\\vartheta_2", "\\vartheta_3", "\\gamma_1", "\\gamma_2")

for (m in stepsizes)
{
    if (m==10) cat("\\begin{tabular}{l || cc | cc | cc | cc | cc| c | c}\\toprule\n") 
  cat("m = ", m, " &")
    for (i in 1:5)
    {
    cat("$",params[i],"$ & & ")
    }    
    cat("ESS & K\\\\\n\\cmidrule{1-13}\n")
    for (p in 1:4)
    {  
            simname <- paste(methods[p], m, sep="")
            thetas <- read.csv(paste("output", psep, simname, psep, "params",".txt",sep=""),header=TRUE, sep=" ")
            thetas["n"]=c()
            trueparam = (read.csv(paste("output", psep, simname, psep, "truth",".txt",sep=""),header=TRUE, sep=" "))
            
            rels <- unlist(colMeans(thetas)/trueparam)
            eff <- effectiveSize(thetas)
            cat(methodabbr[p], " & ")
            for (i in 1:5)
            {    cat(round(rels[i],2)," & ")
                 cat(round(eff,0)[i]," & ")
            }
            
            
            cat( " ", round(multiESS(thetas),2), " &")
            
            cat(nrow(thetas), "\\\\ ")
            
            

            
            #cat(sqrt(sum((apply(thetas,2,mean,na.rm=FALSE) - trueparam)^2)))
            cat("\n")

    }
    
  if (m==100) cat("\\end{tabular}\n")
    
}


 
#Alg.\ 3 &Mean & Standard dev. & Autocorrelation time\\

#$\theta_1$ & 0.0976 & 0.0289 & 97.23 (54\%) \\ 
#$\theta_2$ & 0.7664 & 0.2268 & 100.29 (57\%) \\ 
#$\theta_3$ & 0.3811 & 0.1039 & 20.44 (12\%) \\ 
#$\theta_4$ & 0.2343 & 0.0523 & 15.47 (21\%) \\ 
#$\theta_5$ & 0.0731 & 0.0215 & 127.90 (68\%) \\ 
#$\theta_6$ & 0.7048 & 0.2032 & 121.05 (72\%) \\ 
#$\theta_7$ & 0.3130 & 0.0845 & 21.30 (13\%) \\ 
#$\theta_8$ & 0.1376 & 0.0295 & 15.74 (18\%) \\ 
#\end{tabular}