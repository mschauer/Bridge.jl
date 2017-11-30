using PyPlot, Bridge, Colors
import PyPlot: PyObject
using Bridge: runmean

#simname = "sumgamma"
oneplot = false
withbeta = transdim
bstr = ["", "b"][1 + withbeta]
params = readdlm(joinpath("output", simname, "params.txt"), Float64; skipstart=1)[:,2:end];

withrho1 = rhosigma[1] > 0


function PyObject(t::Color)
    trgb = convert(RGB, t)
    ctup = map(float, (red(trgb), green(trgb), blue(trgb)))
    o = PyPlot.PyObject(ctup)
    return o
end
cd = 40
trqs(x) = 1-sqrt(1-x)

i = 1
dcol1 = PyObject(RGB(trqs.(Bridge._viridis[cd*i])...))
col1 = PyObject(RGB((Bridge._viridis[cd*i])...))
lcol1 = PyObject(RGB(sqrt.(Bridge._viridis[cd*i])...))
i = 2
dcol2 = PyObject(RGB(trqs.(Bridge._viridis[cd*i])...))
col2 = PyObject(RGB((Bridge._viridis[cd*i])...))
lcol2 = PyObject(RGB(sqrt.(Bridge._viridis[cd*i])...))
i = 3
dcol3 = PyObject(RGB(trqs.(Bridge._viridis[cd*i])...))
col3 = PyObject(RGB((Bridge._viridis[cd*i])...))
lcol3 = PyObject(RGB(sqrt.(Bridge._viridis[cd*i])...))

dcol, col, lcol = dcol2, col2, lcol3

if simid == 2
    alpha1 = 2.0
    alpha2 = alpha1/10
    beta1 = 0.4
    beta2 = beta1/10
end
#=
beta = beta1 + beta2

alpha = (beta1*alpha1 + beta2*alpha2)/(beta1 + beta2)
@show alpha
=#

if oneplot

    figure(figsize=(8,5))
    subplot(231)
else
    figure(figsize=(8,5))
    subplot(121)
end        

@show n = size(params, 1)
N = div(size(params, 2)-2,2)
skip = max(iterations÷1_000, 1)



plot(params[:,1], color=lcol , lw = 0.5, label=latexstring("\\alpha"))
abar = runmean(params[:,1])
plot(abar, color=dcol , lw = 0.5, label=latexstring("\\bar\\alpha"))


#=
annotate(latexstring("\\alpha"),
    xy=[div(2n, 3); params[1+div(2n, 3), 1]],
    xytext=[0, -10],
    textcoords="offset points",
    fontsize=10.0,
    ha="right",
    va="bottom")
annotate(latexstring("\\bar\\alpha"),
    xy=[n; abar[end]],
    xytext=[10,0],
    textcoords="offset points",
    fontsize=10.0,
    ha="right",
    va="bottom")=#

plot(1:n, fill(alpha0, n), ":", color=:darkorange , lw = 1.0, label="⌀α")
legend()
grid(linestyle=":", axis="y")

if withbeta
    if oneplot
        subplot(234)
    else 
        subplot(122)
    end    
end    

plot(params[:,2], color=lcol , lw = 0.5, label=latexstring("\\beta"))
bbar = runmean(params[:,2])
plot(bbar, color=dcol , lw = 0.5, label=latexstring("\\bar\\beta"))

#=
annotate(latexstring("\\beta"),
    xy=[div(2n, 3); params[1+div(2n, 3), 2]],
    xytext=[0, -10],
    textcoords="offset points",
    fontsize=10.0,
    ha="right",
    va="bottom")
annotate(latexstring("\\bar\\beta"),
    xy=[n; bbar[end]],
    xytext=[10,0],
    textcoords="offset points",
    fontsize=10.0,
    ha="right",
    va="bottom")
=#
if simid == 2
    plot(1:n, fill(beta0, n), ":", color=:darkorange , lw = 1.0, label="β1+β2")
elseif simid ==3
    plot(1:n, fill(beta0, n), ":", color=:darkorange , lw = 1.0, label=L"\hat\beta")
end
legend()
grid(linestyle=":", axis="y")
    
if oneplot 
    subplot(132)
else
    savefig(joinpath("output", simname, "traceplot1$(bstr).pdf"))
    figure()
    subplot(121)
end

for i in 1:N
    dcol = PyObject(RGB(trqs.(Bridge._viridis[cd*i])...))
    col = PyObject(RGB((Bridge._viridis[cd*i])...))
    lcol = PyObject(RGB(sqrt.(Bridge._viridis[cd*i])...))
    
    plot(params[:, 2 + i], color=lcol, lw = 0.2)
    thbar = runmean(params[:, 2 + i])
    plot(thbar, color=col , lw = 0.5)
    annotate(latexstring("\\theta_$i"),
        xy=[div(2n, 3); params[div(2n, 3), 2 + i]],
        xytext=[0, -10],
        textcoords="offset points",
        fontsize=10.0,
        ha="right",
        va="bottom")
    annotate(latexstring("\\bar\\theta_$i"),
        xy=[n; thbar[end]],
        xytext=[10,0],
        textcoords="offset points",
        fontsize=10.0,
        ha="right",
        va="bottom")
end    

legend()
grid(linestyle=":", axis="y")

if oneplot
    subplot(133)
else
    subplot(122)
end

for i in (2-withrho1):N
    dcol = PyObject(RGB(trqs.(Bridge._viridis[cd*i])...))
    col = PyObject(RGB((Bridge._viridis[cd*i])...))
    lcol = PyObject(RGB(sqrt.(Bridge._viridis[cd*i])...))
    
    plot(params[:, 2+N + i], color = lcol, lw = 0.2)
    rbar = runmean(params[:, 2+N + i])
    plot(rbar, color = col, lw = 0.5)
    annotate(latexstring("\\rho_$i"),
        xy=[div(2n, 3); params[div(2n, 3), 2+N+i]],
        xytext=[0, -10],
        textcoords="offset points",
        fontsize=10.0,
        ha="right",
        va="bottom")
    annotate(latexstring("\\bar\\rho_$i"),
        xy=[n; rbar[end]],
        xytext=[0,0],
        textcoords="offset points",
        fontsize=10.0,
        ha="right",
        va="bottom")
end
grid(linestyle=":", axis="y")

savefig(joinpath("output", simname, "traceplot2$(bstr).pdf"))

truth = readdlm(joinpath("output", simname, "truth.txt"), header=true)

b = truth[1][6:6+N-1] # bin boundaries
phat = mean(params, 1)


function θ(x, params, b)
    N = length(b)
    N == 0 && return 0.0
    x <= b[1] && return 0.0
    x > b[N] && return params[2 + N] * x + params[end]
    k = first(searchsorted(b, x)) - 1
    params[2 + k] * x + params[2 + N + k]
end


theta0(x, alpha1, beta1, alpha2, beta2) = -log((beta1*exp(-alpha1*x) + beta2*exp(-alpha2*x))/(beta1+beta2)) - (beta1*alpha1 + beta2*alpha2)/(beta1 + beta2)*x
phi0(x, alpha1, beta1, alpha2, beta2) = -log((beta1*exp(-alpha1*x) + beta2*exp(-alpha2*x))/(beta1+beta2)) 

xx = 2*(0.01:0.01:2)*b[end]


thetahat(xx, p) = [θ(x, p, b) for x in xx]
logxvhat(xx, p) = [θ(x, p, b) + p[1]*x - log(p[2]) for x in xx]
logxv0(x, alpha1, beta1, alpha2, beta2) = -log((beta1*exp(-alpha1*x) + beta2*exp(-alpha2*x)))
logxv0(x, alpha1, beta1) = -log(beta1*exp(-alpha1*x))


#theta2(x) = θ(x, phat + theta[1] - alpha, b) + alpha*x

if !withbeta

figure(figsize=(8,5))
subplot(121)
for i in 1:40
    j = rand(1:n)
    plot(xx, thetahat(xx, params[j, :])./xx + params[j,1], ":", color=:black, lw=0.2)
end
plot(xx, theta0.(xx, alpha1, beta1, alpha2, beta2)./xx + alpha, label="true")
plot(xx, thetahat(xx, phat)./xx + phat[1], label="θ(x)/x + α")
plot(b, theta0.(b,  alpha1, beta1, alpha2, beta2)./b + alpha, "o")

#plot(xx, thetahat(xx, phat)./xx + phat[1] - alpha, label="θ(x)/x - C", ":", color=:orange)
legend()

subplot(122)
for i in 1:40
    j = rand(1:n)
    plot(xx, thetahat(xx, params[j, :]) + params[j, 1]*xx, ":", color=:black, lw=0.2)
end
    
plot(xx, phi0.(xx, alpha1, beta1, alpha2, beta2), label="true")
plot(xx, thetahat(xx, phat) + phat[1]*xx - log(phat[2]), label = "αx + θ(x) - log β" )
plot(b, phi0.(b,  alpha1, beta1, alpha2, beta2), "o")

legend()

end # if 

if false

    figure()

        
    plot(xx, phi0.(xx, alpha1, beta1, alpha2, beta2), label="true")
    plot(xx, thetahat(xx, phat) + phat[1]*xx + log(phat[2]), label = "αx + θ(x) + log β" )
    plot(b, phi0.(b,  alpha1, beta1, alpha2, beta2), "o")

    legend()
end

println([quantile(params[:,1] + params[:,4], q) for q in [0.05, 0.5, 0.95]])
n1 = div(n,2)
n2 = n

A = zeros(length(xx), length(n1:skip:n2))
i_ = 1
for i in n1:skip:n2
    A[:, i_] = logxvhat(xx, params[i, :])
    i_ += 1
end


upper = mapslices(v-> quantile(v,0.975), A, 2)
med = median(A, 2)
lower = mapslices(v-> quantile(v,0.025), A, 2)

figure(figsize=(8,5))
#subplot(121)
fill_between(xx, upper[:], lower[:], edgecolor=lcol2, facecolor=(lcol2..., 0.2), hatch="X", label="95% cred. band")
#plot(xx, med, label = "marg. post. median", color=:blue)
if simid == 2
    plot(xx, logxv0.(xx, alpha1, beta1, alpha2, beta2), label=L"-\log(x v_0(x))", color=:darkorange)
elseif simid == 3
    plot(xx, logxv0.(xx, alpha0, beta0), label=L"-\log(x \hat v(x))", color=:darkorange)
end  
legend()
savefig(joinpath("output", simname, "bands$(bstr)1.pdf"))
#savefig(joinpath("output", simname, "bands$(bstr)1.svg"))

A2 = zeros(length(xx), n2 - n1 + 1)
for i in n1:skip:n2
    A2[:, i - n1 + 1] = logxvhat(xx, params[i, :])./xx 
end


upper = mapslices(v-> quantile(v,0.975), A2, 2)
med = median(A2, 2)
lower = mapslices(v-> quantile(v,0.025), A2, 2)

if false
figure(figsize=(8,5))
#subplot(122)
fill_between(xx, upper[:], lower[:], color=lcol2, hatch="X", label="95% cred. band")
#plot(xx, med, label = "marg. post. median", color=:blue)
plot(xx, logxv0.(xx, alpha1, beta1, alpha2, beta2)./xx, label=L"(-\log (x v_0(x))/x", color=:darkorange)
legend()

savefig(joinpath("output", simname, "bands$(bstr)2.pdf"))

end
#clf();[display(plot(params[:, i], label="$i", linewidth=0.4)) for i in 1:2N+1];legend();
