pgfplots()
srand(3)
a1 = 1.0
H = 50
for f in 1:3
    a = 5*[2.5, 3.0, 3.0][f]
    aζ = 5*[3.0, 2.5, 3.0][f]
    lim = [22.0, 1.05, 2.0][f]
    name = "output/volatility/figprior$f.pdf"

    v = zeros(H)
    ζ = zeros(H)

    v[1] = rand(InverseGamma(a1, a1))

    for i in 2:H
        ζ[i] = rand(InverseGamma(aζ, inv(v[i-1]/aζ)))
        v[i] = rand(InverseGamma(a, inv(ζ[i]/a)))
    end

    display(plot(v, color=colorant"#0044FF", size = (300, 300/1.4), ylim=(0.0, lim), label = LaTeXString("\$\\alpha = $a, \\alpha_\\zeta = $aζ\$")))
    savefig(name)
    println(v)
end