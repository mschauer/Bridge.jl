clf();
xx = 0.01:0.01:5
plot(xx, phi0.(xx, alpha1, beta1, alpha2, beta2), color=col1, label=L"-\log (xv(x))")
plot(xx, alpha*xx, label=L"\alpha x", color=col2)
plot(xx, alpha2*xx - alpha2*xx[end] + phi0.(xx[end], alpha1, beta1, alpha2, beta2), color=col3, label=L"\alpha_2x - \operatorname{const}");
legend()