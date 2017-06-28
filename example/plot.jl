
if PLOT == :winston

using Winston
import Winston: plot, oplot
function plot(Y::SamplePath{SVector{2,Float64}}, args...; keyargs...) 
    yy = Bridge.mat(Y.yy)
    plot(yy[1,:], yy[2,:], args...; keyargs...)
end    
function plot(Y::SamplePath{SVector{1,Float64}}, args...; keyargs...) 
    yy = Bridge.mat(Y.yy)
    plot(Y.tt, yy[1,:], args...; keyargs...)
end    
function plot(Y::SamplePath{Float64}, args...; keyargs...) 
    plot(Y.tt, Y.yy, args...; keyargs...)
end    

function oplot(Y::SamplePath{SVector{1,Float64}}, args...; keyargs...) 
    yy = Bridge.mat(Y.yy)
    oplot(Y.tt, yy[1,:], args...; keyargs...)
end   
function oplot(Y::SamplePath{SVector{2,Float64}}, args...; keyargs...) 
    yy = Bridge.mat(Y.yy)
    oplot(yy[1,:], yy[2,:], args...; keyargs...)
end    
function oplot(Y::SamplePath{Float64}, args...; keyargs...) 
    oplot(Y.tt, Y.yy, args...; keyargs...)
end    

function oplot2(Y::SamplePath{SVector{2,Float64}},  a1="r", a2="b";  keyargs...) 
    yy = Bridge.mat(Y.yy)
    oplot(Y.tt, yy[1,:], a1; keyargs...)
    oplot(Y.tt, yy[2,:], a2; keyargs...)
end    
function plot2(Y::SamplePath{SVector{2,Float64}}, a1="r", a2="b"; keyargs...) 
    yy = Bridge.mat(Y.yy)
    plot(Y.tt,  yy[1,:], a1; keyargs...)
    oplot(Y.tt, yy[2,:], a2; keyargs...)
end    

elseif PLOT == :pyplot

using PyPlot
import PyPlot: plot
 
function plot(Y::SamplePath{Float64}; keyargs...) 
    plot(Y.tt, Y.yy; keyargs...)
end    
function plot(Y::SamplePath{SVector{2,Float64}}; keyargs...) 
    yy = Bridge.mat(Y.yy)
    plot(yy[1,:], yy[2,:]; keyargs...)
end

end


# ---
