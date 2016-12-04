function res = run_deepdcf(subS, rp, bSaveImage)
init_rect = subS.init_rect;
img_files = subS.s_frames;
time = 0;
results = repmat(init_rect,[numel(img_files),1]);

%% setting
useGPU = ispc();
visualization = true;

net = load('simplenn_vgg_deepdcfnet.mat');
net = vl_simplenn_tidy(net.net);
if useGPU    %gpuSupport
    net = vl_simplenn_move(net, 'gpu');
end

normalized_sz = net.meta.normalization.imageSize(1:2);
averageImage = net.meta.normalization.averageImage;
lambda = 1e-4;
padding = 1.5;
output_sigma_factor = 0.1;
interp_factor = 0.02;
output_sigma = sqrt(prod([50,50]))*output_sigma_factor;

yf = fft2(gaussian_shaped_labels(output_sigma, normalized_sz));
if useGPU, yf = gpuArray(yf);end    %gpuSupport

pos = floor(init_rect([2,1])+init_rect([4,3])/2);
target_sz = init_rect([4,3]);
window_sz = floor(target_sz*(1+padding));

for frame = 1:numel(img_files)
    image = imread(img_files{frame});
    if size(image,3)==1
        image = repmat(image,[1,1,3]);
    end
    tic
    if frame >1
        patch_crop = single(imresize(get_subwindow(image, pos, window_sz),...
            normalized_sz));
        
        if useGPU,patch_crop= gpuArray(patch_crop);end    %gpuSupport
        
        search = bsxfun(@minus,patch_crop, averageImage);
        
        res = vl_simplenn(net, search,[],[],'mode','test','conserveMemory',true);
        zf = fft2(res(end).x);
        
        kzf = sum(zf.*conj(model_xf),3)/numel(zf);
        
        response = real(ifft2(model_alphaf .* kzf));
        [vert_delta, horiz_delta] = find(response == max(response(:)), 1);
        if vert_delta > size(zf,1) / 2,  %wrap around to negative half-space of vertical axis
            vert_delta = vert_delta - size(zf,1);
        end
        if horiz_delta > size(zf,2) / 2,  %same for horizontal axis
            horiz_delta = horiz_delta - size(zf,2);
        end
        pos = pos + [vert_delta - 1, horiz_delta - 1].*...
            window_sz./normalized_sz;
    end
    
    patch = single(imresize(get_subwindow(image, pos, window_sz),normalized_sz));
    if useGPU, patch= gpuArray(patch);end    %gpuSupport
    
    target = bsxfun(@minus,patch,averageImage);
    res = vl_simplenn(net,target,[],[],'mode','test','conserveMemory',true);
    xf = fft2(res(end).x);
    
    kf = sum(xf.*conj(xf),3)/numel(xf);
    alphaf = yf ./ (kf + lambda);
    if frame == 1
        model_alphaf = alphaf;
        model_xf = xf;
    else
        model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;
        model_xf = (1 - interp_factor) * model_xf + interp_factor * xf;
    end
    box = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
    results(frame,:) = gather(box);
    time = time + toc;
    if visualization
        if frame >1
            subplot(1,2,1);im_show_add_response(patch_crop,response);
        end
        subplot(1,2,2);imshow(uint8(image));
        rectangle('Position',results(frame,:),'EdgeColor','g');
        drawnow;
    end
end

res.type = 'rect';
res.res = results;
res.fps    = numel(img_files) / time;
end


function im_show_add_response(im,response)
sz = size(response);
response = circshift(response, floor(sz(1:2) / 2) - 1);

imshow(uint8(gather(im)));hold on;
h = imagesc(response);colormap(jet);
set(h,'AlphaData',gather(response)+0.35);
end


function out = get_subwindow(im, pos, sz)
if isscalar(sz),  %square sub-window
    sz = [sz, sz];
end
xs = floor(pos(2)) + (1:sz(2)) - floor(sz(2)/2);
ys = floor(pos(1)) + (1:sz(1)) - floor(sz(1)/2);
xs(xs < 1) = 1;
ys(ys < 1) = 1;
xs(xs > size(im,2)) = size(im,2);
ys(ys > size(im,1)) = size(im,1);
out = im(ys, xs, :);
end


function labels = gaussian_shaped_labels(sigma, sz)
[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
labels = exp(-0.5 / sigma^2 * (rs.^2 + cs.^2));
labels = circshift(labels, -floor(sz(1:2) / 2) + 1);
assert(labels(1,1) == 1)
end