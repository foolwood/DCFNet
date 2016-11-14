xf = rand(125,125,256,16,'single')+1j*rand(125,125,256,16,'single');
zf = rand(125,125,256,16,'single')+1j*rand(125,125,256,16,'single');
obj.yf = rand(125,125,1,16,'single')+1j*rand(125,125,1,16,'single');
obj.lambda = 1e-4;
xf_conj = conj(xf);
[h,w,c,~] = size(xf);
mn = h*w*c;
            
tic
for i = 1:20
    r1 = xf./obj.lambda;

end
toc

tic
for i = 1:20
    r2 = xf/obj.lambda;
end
toc

sum(r1(:)-r2(:))

