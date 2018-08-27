using PyPlot, Bridge, Colors
import PyPlot: PyObject
using Bridge: runmean

#simname = "sumgamma"
oneplot = false
withbeta = transdim
bstr = ["", "b"][1 + withbeta]
params = readdlm(joinpath("output", simname, "params.txt"), Float64; skipstart=1)[:,2:end];

withrho1 = N > 0 && rhosigma[1] > 0


function PyObject(t::Color)
    trgb = convert(RGB, t)
    ctup = map(float, (red(trgb), green(trgb), blue(trgb)))
    o = PyPlot.PyObject(ctup)
    return o
end
cd = 40
cd = 30
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
######## alpha and beta

if oneplot

    figure(figsize=(8,5))
    subplot(231)
else 
    if withbeta
        figure(figsize=(8,5))
        subplot(121)
    else
        figure(figsize=(4,5))
    end
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

plot(1:n, fill(alpha0, n), ":", color=:darkorange , lw = 1.0, label=L"\varnothing\alpha")
legend()
grid(linestyle=":", axis="y")

if withbeta
    if oneplot
        subplot(234)
    else 
        subplot(122)
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
        plot(1:n, fill(beta0, n), ":", color=:darkorange , lw = 1.0, label=L"\beta_1+\beta_2")
    elseif simid ==3
        plot(1:n, fill(beta0, n), ":", color=:darkorange , lw = 1.0, label=L"\hat\beta")
    end
    legend()
    grid(linestyle=":", axis="y")
    
end

if oneplot 
    subplot(132)
else
    savefig(joinpath("output", simname, "traceplot1$(bstr).pdf"))
    figure()
    subplot(121)
end

######## theta and rho (added or not)

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

#legend()
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

if N > 0
xx = (0.002:0.01:2)*b[end]
else 
    xx = (0.002:0.01:8)
end

thetahat(xx, p) = [θ(x, p, b) for x in xx]
logxvhat(xx, p) = [θ(x, p, b) + p[1]*x - log(p[2]) for x in xx]
logxv0(x, alpha1, beta1, alpha2, beta2) = -log((beta1*exp(-alpha1*x) + beta2*exp(-alpha2*x)))
logxv0(x, alpha1, beta1) = -log(beta1) + alpha1*x


#theta2(x) = θ(x, phat + theta[1] - alpha, b) + alpha*x

#println([quantile(params[:,1] + params[:,4], q) for q in [0.05, 0.5, 0.95]])
n1 = div(n, 4)
#n1 = 20000
n2 = n
phat = median(params[n1:n2,:], 1)

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
    sa, sb = [0.0376041, 0.1005920]*1.96

    plot(xx, logxv0.(xx,  0.5889420, 1.8127392 *365/7 ), label=L"-\log(x \hat v(x))", color=:darkorange)
  #  fill_between(xx,  logxv0.(xx, 0.5889420+sa,(1.8127392 -sb)*365/7), logxv0.(xx, 0.5889420-sa ,(1.8127392 + sb)*365/7), edgecolor=:orange, facecolor=:None, label="freq. band")
    #plot(xx, logxv0.(xx, (phat[1] + phat[end-N]), (phat[2] * exp(-phat[end]))))
end  
legend()
savefig(joinpath("output", simname, "bands$(bstr)1.pdf"))
#savefig(joinpath("output", simname, "bands$(bstr)1.svg"))

z = sort(diff(X.yy))
figure();plot(z, range(0, stop=1, length=length(z)))
#plot(z, cdf.(Gamma( var(z)/mean(z), mean(z)^2/var(z)), z))
plot(z, cdf.(Gamma(beta0*dt, 1/alpha), z), label="Gamma ML")
plot(z, cdf.(Gamma(phat[2]*dt, 1/phat[1]), z), label="Gamma(beta,alpha)")
#plot(z, cdf.(Gamma(params[end,2]*dt, 1/params[end,1]), z), label="Gamma(beta,alpha)")
N > 1 && try
    plot(z, cdf.(Gamma((phat[2] * exp(-phat[end-1]))*dt, 1/(phat[1] + phat[end-N-1])), z), label="Gamma mid")
catch
end
plot(z, cdf.(Gamma((phat[2] * exp(-phat[end]))*dt, 1/(phat[1] + phat[end-N])), z), label="Gamma end")
simid==2 && axis([0.0, 6.0, 0.4, 1.0])
legend()
#=
figure();plot(z, linspace(0,1,length(z))-cdf.(Gamma(  var(z)/mean(z), mean(z)^2/var(z)), z))
=#

#=
betahat = var(z)/mean(z)/dt
alphahat = 1/(mean(z)^2/var(z))

smooth(z, hw, D = Normal()) = conv(z, normalize(pdf.(D, linspace(-std(D)*3,std(D)*3, 2*hw)), 1))[hw:end-hw]
sz = smooth(z, 20)[1:end-20]
figure();plot(z[2:end-20], 1/length(z)./diff(sz), linewidth=0.2)
G = Gamma(  var(z)/mean(z), mean(z)^2/var(z))
plot(z, pdf.(G, z))
=#
#=
Plot Levy density of end piece
plot(xx, -llevyx.(GammaProcess((phat[2] * exp(-phat[end])), (phat[1] + phat[end-N])), xx))
=#


z = sort(diff(X.yy))
figure();plot(z, range(0, stop=1, length=length(z)))
plot(z, cdf.(Gamma(beta0*dt, 1/alpha), z), label="Gamma ML")
plot(z, cdf.(Gamma(phat[2]*dt, 1/phat[1]), z), label="Gamma(beta,alpha)")
if N > 0
    i = searchsortedfirst(z, b[1]/2)
else 
    i = length(z)÷2
end
axis([0.0, z[i], 0, i/length(z)])
legend()

ps = mean(diff(params,1).!=0,1) 
ps *= 5
ps[2] /= 4
ps