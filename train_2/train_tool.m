close all
zf = fft2(z);
sigma = 5;
sz = [125,125];
[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
labels = exp(-0.5 / sigma^2 * (rs.^2 + cs.^2));
labels = circshift(labels, -floor(sz(1:2) / 2) + 1);
labels = circshift(labels, [6,-10]);


dldzf = -bsxfun(@times,dldrf.*alphaf,xf_conj)/mn;

% dldxf = bsxfun(@times,dldrf,...
%     bsxfun(@rdivide,...
%     bsxfun(@times,bsxfun(@times,zf,obj.yf)/mn,kxxf+obj.lambda)-...
%     bsxfun(@times,bsxfun(@times,(sum(zf.*xf_conj,3)/mn),obj.yf),xf/mn),...
%     (kxxf + obj.lambda).*(kxxf + obj.lambda)));


dldxf = bsxfun(@times,dldrf,...
    bsxfun(@rdivide,...
    -1*bsxfun(@times,bsxfun(@times,zf,obj.yf)/mn,kxxf+obj.lambda)+...
    1*bsxfun(@times,bsxfun(@times,(sum(zf.*xf_conj,3)/mn),obj.yf),xf/mn),...
    (kxxf + obj.lambda).*(kxxf + obj.lambda)));




zf0 = zf-dldzf*0;
kz0xf = sum(zf0 .* xf_conj, 3) ./ mn;
r0 = real(ifft2(alphaf .* kz0xf));
% imagesc(r1);colorbar();

zf1 = zf-dldzf*1;
kz1xf = sum(zf1 .* xf_conj, 3) ./ mn;
r1 = real(ifft2(alphaf .* kz1xf));
% imagesc(r1);colorbar();

delta_r = r1-r0;
subplot(1,2,1);imagesc(delta_r);colorbar();
hold on;
plot(116,7,'r*');


xf1 = xf+dldxf*2;
kzx1f = sum(zf .* conj(xf1), 3) ./ mn;
kx1x1f = sum(xf1 .* conj(xf1), 3) / mn;
alphaf1 = bsxfun(@rdivide,obj.yf,(kx1x1f + obj.lambda));
r2 = real(ifft2(alphaf1 .* kzx1f));

delta_r = r2-r0;
subplot(1,2,2);imagesc(delta_r);colorbar();
hold on;
plot(116,7,'r*');