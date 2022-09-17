function [rec] = process(x,y,dir)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
filename=sprintf("%d%02d%02d",dir,x,y);
rec=[];
a=read_complex_binary2(filename);
num_frame=round(size(a,1)/65536)-2;
for i=1:num_frame
section=a(i*65536+1:(i+1)*65536);
spectrum=db(fftshift(fft(section)));
rec(i)=max(spectrum(22909:22969));
end
end

