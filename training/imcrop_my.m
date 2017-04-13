function img_crop = imcrop_my(img,bbox,padding,sz)
xi = linspace(-1, 1, sz(2));
yi = linspace(-1, 1, sz(1));
[xx,yy] = meshgrid(xi,yi);
yyxx = single([yy(:), xx(:)]') ; % 2xM
[im_h,im_w,im_c,~] = size(img);
if im_c == 1
    img = repmat(img,[1,1,3,1]);
end

target_crop_w = (1+padding)*(bbox(:,3)-bbox(:,1));
target_crop_h = (1+padding)*(bbox(:,4)-bbox(:,2));
target_crop_cx = (bbox(:,1)+bbox(:,3))/2;
target_crop_cy = (bbox(:,2)+bbox(:,4))/2;

cy_t = (target_crop_cy*2/(im_h-1))-1;
cx_t = (target_crop_cx*2/(im_w-1))-1;

h_s = target_crop_h/(im_h-1);
w_s = target_crop_w/(im_w-1);

s = reshape([h_s,w_s]', 2, 1, []); % x,y scaling
t = reshape([cy_t,cx_t]', 2, 1, []); % translation

g = bsxfun(@times, yyxx, s); % scale
g = bsxfun(@plus, g, t); % translate
g = reshape(g, 2, sz(1), sz(2), []);

img_crop = vl_nnbilinearsampler(img, g);

end