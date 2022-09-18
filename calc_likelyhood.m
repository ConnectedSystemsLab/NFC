function [res] = calc_likelyhood(f_mean,f_dev,rssi,n,x,y,lambda)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
res=1;
res=res*normpdf(rssi(1),f_mean(x,y)+lambda,f_dev(x,y));
res=res*normpdf(rssi(2),f_mean(n-y,x)+lambda,f_dev(n-y,x));
res=res*normpdf(rssi(3),f_mean(n-x,n-y)+lambda,f_dev(n-x,n-y));
res=res*normpdf(rssi(4),f_mean(y,n-x)+lambda,f_dev(y,n-x));
end

