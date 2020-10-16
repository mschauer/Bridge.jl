pgfplots()

a1 = .6
H = 400
for f in 1:3
    srand(3)
    a = 10*[2.0, 4.0, 3.0][f]
    aζ = 10*[4.0, 2.0, 3.0][f]
   
    lim = (1e-4, 1e10)
#    a1 = a
 
    name = "output/volatility/figprior$f.pdf"
    sn = 5
    local p
    v = zeros(H)
    ζ = zeros(H)


    for s in 1:sn
        v[1] = rand(InverseGamma(a1, a1))
        v[1] = 500.0
        for i in 2:H
            ζ[i] = rand(InverseGamma(aζ, inv(v[i-1]/aζ)))
            v[i] = rand(InverseGamma(a, inv(ζ[i]/a)))
        end
        c = RGB( Bridge._viridis[25*s]...)
        if s == 1

            p = plot(v, yscale = :log10, color=colorant"#0044FF", size = (300, 300/1.4),  ylim=lim,  color = c, width= 0.8, legend=false, title = LaTeXString("\$\\alpha = $a, \\alpha_\\zeta = $aζ\$"))
        else
            p = plot!(v, yscale = :log10, color=colorant"#0044FF", size = (300, 300/1.4), #=, ylim=lim, =# color = c, width=0.8)
        end
        
    end   
    display(p)   
    
    savefig(name)
    println(v)
end