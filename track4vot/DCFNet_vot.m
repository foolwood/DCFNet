function DCFNet_vot(varargin)

% *************************************************************
% VOT: Always call exit command at the end to terminate Matlab!
% *************************************************************
cleanup = onCleanup(@() exit() );

% *************************************************************
% VOT: Set random seed to a different value every time.
% *************************************************************
RandStream.setGlobalStream(RandStream('mt19937ar', 'Seed', sum(clock)));

% **********************************
% VOT: Get initialization data
% **********************************
[handle, image, region] = vot('rectangle');
gpuDevice(randi(2,1));
vl_setupnn();
% Initialize the tracker
im = vl_imreadjpeg({image});
[state, ~] = DCFNet_initialize(im{1}, region, varargin);

while true
    
    % **********************************
    % VOT: Get next frame
    % **********************************
    [handle, image] = handle.frame(handle);
    
    if isempty(image)
        break;
    end;
    
    % Perform a tracking step, obtain new region
    im = vl_imreadjpeg({image});
    [state, region] = DCFNet_update(state, im{1});
    
    % **********************************
    % VOT: Report position for frame
    % **********************************
    handle = handle.report(handle, region);
    
end;

% **********************************
% VOT: Output the results
% **********************************
handle.quit(handle);

end

function [state, location] = DCFNet_initialize(I, region, varargin)
state.gpu = true;
state.visual = false;

state.lambda = 1e-4;
state.padding = 1.5;
state.output_sigma_factor = 0.1;
state.interp_factor = 0.023;

state.numScale = 3;
state.scaleStep = 1.0375;
state.min_scale_factor = 0.2;
state.max_scale_factor = 5;
state.scale_penalty = 0.9745;
state.net_index = 6;
state = vl_argparse(state, varargin{1,1});

net_name = ['DCFNet-', num2str(state.net_index),'.mat'];
net = load(net_name);
net = vl_simplenn_tidy(net.net);
if state.gpu    %gpuSupport
    net = vl_simplenn_move(net, 'gpu');
    I = gpuArray(I);
end
state.net = net;

state.scale_factor = state.scaleStep.^((1:state.numScale)-ceil(state.numScale/2));
state.scalePenalty = ones(1,state.numScale);
state.scalePenalty((1:state.numScale)~=ceil(state.numScale/2)) = state.scale_penalty;

state.norm_size = state.net.meta.normalization.imageSize(1:2);

state.output_sigma = sqrt(prod([50,50]))*state.output_sigma_factor;
state.yf = single(fft2(gaussian_shaped_labels(state.output_sigma, state.norm_size)));
if state.gpu, state.yf = gpuArray(state.yf);end    %gpuSupport
state.cos_window = single(hann(size(state.yf,1)) * hann(size(state.yf,2))');
if state.gpu, state.cos_window = gpuArray(state.cos_window);end    %gpuSupport

state.pos = region([2,1])+region([4,3])/2;
state.target_sz = region([4,3])';
state.min_sz = max(4,state.min_scale_factor.*state.target_sz);
state.max_sz = state.max_scale_factor.*state.target_sz;

window_sz = state.target_sz*(1+state.padding);
patch = imcrop_multiscale(I, state.pos, window_sz, state.norm_size);
if state.gpu,patch= gpuArray(patch);end    %gpuSupport
target = bsxfun(@minus,patch,state.net.meta.normalization.averageImage);
res = vl_simplenn(state.net, target);

xf = fft2(bsxfun(@times, res(end).x, state.cos_window));
state.numelxf = numel(xf);
kf = sum(xf.*conj(xf),3)/state.numelxf;
state.model_alphaf = state.yf ./ (kf + state.lambda);
state.model_xf = xf;

location = region;
if state.visual
    subplot(1,2,1);imshow(uint8(patch));
    subplot(1,2,2);imshow(uint8(I));
    rectangle('Position',location,'EdgeColor','g');
    drawnow;
end

end

function [state, location] = DCFNet_update(state, I, varargin)
if state.gpu, I = gpuArray(I);end
window_sz = bsxfun(@times, state.target_sz, state.scale_factor)*(1+state.padding);
patch_crop = imcrop_multiscale(I, state.pos, window_sz, state.norm_size);
if state.gpu,patch_crop= gpuArray(patch_crop);end    %gpuSupport
search = bsxfun(@minus,patch_crop,state.net.meta.normalization.averageImage);
res = vl_simplenn(state.net, search);

zf = fft2(bsxfun(@times, res(end).x, state.cos_window));
kzf = sum(bsxfun(@times, zf, conj(state.model_xf)),3)/state.numelxf;

response = squeeze(real(ifft2(bsxfun(@times, state.model_alphaf, kzf))));
[max_response, max_index] = max(reshape(response,[],state.numScale));
max_response = max_response.*state.scalePenalty;
scale_delta = find(max_response == max(max_response),1,'last');
[vert_delta, horiz_delta] = ind2sub(state.norm_size,max_index(scale_delta));

if vert_delta > size(response,1) / 2  %wrap around to negative half-space of vertical axis
    vert_delta = vert_delta - size(response,1);
end
if horiz_delta > size(response,2) / 2  %same for horizontal axis
    horiz_delta = horiz_delta - size(response,2);
end
window_sz = window_sz(:,scale_delta);
state.pos = state.pos + [vert_delta - 1, horiz_delta - 1].*...
    window_sz'./state.norm_size;
state.target_sz = min(max(window_sz/(1+state.padding),state.min_sz),state.max_sz);

patch = imcrop_multiscale(I, state.pos, window_sz, state.norm_size);
if state.gpu, patch= gpuArray(patch);end    %gpuSupport
target = bsxfun(@minus, patch, state.net.meta.normalization.averageImage);

res = vl_simplenn(state.net, target);
xf = fft2(bsxfun(@times, res(end).x, state.cos_window));
kf = sum(xf .* conj(xf), 3) / numel(xf);
alphaf = state.yf ./ (kf + state.lambda);   %equation for fast training

state.model_alphaf = (1 - state.interp_factor) * state.model_alphaf + state.interp_factor * alphaf;
state.model_xf = (1 - state.interp_factor) * state.model_xf + state.interp_factor * xf;

box = [state.pos([2,1]) - state.target_sz([2,1])'/2, state.target_sz([2,1])'];

location = double(gather(box));

if state.visual
    subplot(1,2,1);im_show_add_response(patch_crop(:,:,:,scale_delta), response(:,:,scale_delta));
    subplot(1,2,2);imshow(uint8(I));
    rectangle('Position',location,'EdgeColor','g');
    drawnow;
end

end


function im_show_add_response(im,response)
sz = size(response);
response = circshift(response, floor(sz(1:2) / 2) - 1);

imshow(uint8(gather(im)));hold on;
h = imagesc(response);colormap(jet);
set(h,'AlphaData',gather(response)+0.6);
end

function labels = gaussian_shaped_labels(sigma, sz)
[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
labels = exp(-0.5 / sigma^2 * (rs.^2 + cs.^2));
labels = circshift(labels, -floor(sz(1:2) / 2) + 1);
assert(labels(1,1) == 1)
end

function img_crop = imcrop_multiscale(img, pos, sz, output_sz)
yi = linspace(-1, 1, output_sz(1));
xi = linspace(-1, 1, output_sz(2));
[xx,yy] = meshgrid(xi,yi);
yyxx = single([yy(:), xx(:)]') ; % 2xM
[im_h,im_w,im_c,~] = size(img);

if im_c == 1
    img = repmat(img,[1,1,3,1]);
end

pos = gather(pos);
sz = gather(sz);
im_h = gather(im_h);
im_w = gather(im_w);

cy_t = (pos(1)*2/(im_h-1))-1;
cx_t = (pos(2)*2/(im_w-1))-1;

h_s = sz(1,:)/(im_h-1);
w_s = sz(2,:)/(im_w-1);

s = reshape([h_s;w_s], 2,1,[]); % x,y scaling
t = [cy_t;cx_t]; % translation
if isa(img,'gpuArray')
    yyxx = gpuArray(yyxx);
    s = gpuArray(s);
    t = gpuArray(t);
end
g = bsxfun(@times, yyxx, s); % scale
g = bsxfun(@plus, g, t); % translate
g = reshape(g, 2, output_sz(1), output_sz(2), []);

img_crop = vl_nnbilinearsampler(img, g);
end