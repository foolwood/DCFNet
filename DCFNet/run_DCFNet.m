function res = run_DCFNet(subS, rp, bSaveImage, varargin)

tracking_env();
cur_path = fileparts(mfilename('fullpath'));
net_file = 'DCFNet-net-7-125-2.mat';
net = load(fullfile(cur_path, net_file));
net = vl_simplenn_tidy(net.net);

param = [];
param.gpu = true;
param.visual = false;
param.net = net;
param.interp_factor = net.meta.interp_factor;
param.scale_penalty = net.meta.scale_penalty;
param.scale_step = net.meta.scale_step;
param.padding = net.meta.padding;
param = vl_argparse(param, varargin) ;

tic;
init_rect = subS.init_rect; %1-index
im = vl_imreadjpeg(subS.s_frames, 'numThreads', 12);

num_frame = numel(im);
result = repmat(init_rect, [num_frame, 1]);

init_rect(1:2) = init_rect(1:2)-1; %0-index
[state, ~] = DCFNet_initialize(im{1}, init_rect, param);
for frame = 2:num_frame
    [state, region] = DCFNet_update(state,im{frame});
    region(1:2) = region(1:2)+1; %1-index
    result(frame,:) = region;
end
time = toc;
res.type = 'rect';
res.res = result;
res.fps = num_frame / time;
end

function [state, location] = DCFNet_initialize(I, region, param)
state.gpu = true;
state.visual = false;

state.lambda = 1e-4;
state.padding = 2.0;
state.output_sigma_factor = 0.1;
state.interp_factor = 0.01;

state.num_scale = 3;
state.scale_step = 1.0275;
state.min_scale_factor = 0.2;
state.max_scale_factor = 5;
state.scale_penalty = 0.9925;
state.net = [];
state = vl_argparse(state, param);

state.scale_factor = state.scale_step.^((1:state.num_scale)-ceil(state.num_scale/2));
state.scale_penalties = ones(1,state.num_scale);
state.scale_penalties((1:state.num_scale)~=ceil(state.num_scale/2)) = state.scale_penalty;

state.net_input_size = state.net.meta.normalization.imageSize(1:2);
state.net_average_image = state.net.meta.normalization.averageImage;

output_sigma = sqrt(prod(state.net_input_size./(1+state.padding)))*state.output_sigma_factor;
state.yf = single(fft2(gaussian_shaped_labels(output_sigma, state.net_input_size)));
state.cos_window = single(hann(size(state.yf,1)) * hann(size(state.yf,2))');

yi = linspace(-1, 1, state.net_input_size(1));
xi = linspace(-1, 1, state.net_input_size(2));
[xx,yy] = meshgrid(xi,yi);
state.yyxx = single([yy(:), xx(:)]') ; % 2xM

if state.gpu %gpuSupport
    state.yyxx = gpuArray(state.yyxx);
    state.net = vl_simplenn_move(state.net, 'gpu');
    I = gpuArray(I);
    state.yf = gpuArray(state.yf);
    state.cos_window = gpuArray(state.cos_window);
end

state.pos = region([2,1])+region([4,3])/2;
state.target_sz = region([4,3])';
state.min_sz = max(4,state.min_scale_factor.*state.target_sz);
[im_h,im_w,~] = size(I);
state.max_sz = min([im_h;im_w],state.max_scale_factor.*state.target_sz);

window_sz = state.target_sz*(1+state.padding);
patch = imcrop_multiscale(I, state.pos, window_sz, state.net_input_size, state.yyxx);
target = bsxfun(@minus, patch, state.net_average_image);
res = vl_simplenn(state.net, target);

x = bsxfun(@times, res(end).x, state.cos_window);
xf = fft2(x);
state.numel_xf = numel(xf);
kf = sum(xf.*conj(xf),3)/state.numel_xf;
state.model_alphaf = state.yf ./ (kf + state.lambda);
state.model_xf = xf;

location = region;
if state.visual
    state.videoPlayer = vision.VideoPlayer('Position', [100 100 [size(I,2), size(I,1)]+30]);
end

end

function [state, location] = DCFNet_update(state, I, varargin)
if state.gpu, I = gpuArray(I);end
window_sz = bsxfun(@times, state.target_sz, state.scale_factor)*(1+state.padding);
patch_crop = imcrop_multiscale(I, state.pos, window_sz, state.net_input_size, state.yyxx);
search = bsxfun(@minus, patch_crop, state.net_average_image);
res = vl_simplenn(state.net, search);

z = bsxfun(@times, res(end).x, state.cos_window);
zf = fft2(z);
kzf = sum(bsxfun(@times, zf, conj(state.model_xf)),3)/state.numel_xf;

response = squeeze(real(ifft2(bsxfun(@times, state.model_alphaf, kzf))));
[max_response, max_index] = max(reshape(response,[],state.num_scale));
max_response = gather(max_response);
max_index = gather(max_index);
[~,scale_delta] = max(max_response.*state.scale_penalties);
[vert_delta, horiz_delta] = ind2sub(state.net_input_size, max_index(scale_delta));

if vert_delta > state.net_input_size(1) / 2  %wrap around to negative half-space of vertical axis
    vert_delta = vert_delta - state.net_input_size(1);
end
if horiz_delta > state.net_input_size(2) / 2  %same for horizontal axis
    horiz_delta = horiz_delta - state.net_input_size(2);
end
window_sz = window_sz(:,scale_delta);
state.pos = state.pos + [vert_delta - 1, horiz_delta - 1].*...
    window_sz'./state.net_input_size;
state.target_sz = min(max(window_sz./(1+state.padding), state.min_sz), state.max_sz);

patch = imcrop_multiscale(I, state.pos, window_sz, state.net_input_size, state.yyxx);
target = bsxfun(@minus, patch, state.net_average_image);

res = vl_simplenn(state.net, target);
x = bsxfun(@times, res(end).x, state.cos_window);
xf = fft2(x);
kf = sum(xf .* conj(xf), 3) / state.numel_xf;
alphaf = state.yf ./ (kf + state.lambda);   %equation for fast training

state.model_alphaf = (1 - state.interp_factor) * state.model_alphaf + state.interp_factor * alphaf;
state.model_xf = (1 - state.interp_factor) * state.model_xf + state.interp_factor * xf;

box = [state.pos([2,1]) - state.target_sz([2,1])'/2, state.target_sz([2,1])'];

location = double(gather(box));

if state.visual
    im_show = insertShape(uint8(gather(I)), 'Rectangle', location, 'LineWidth', 4, 'Color', 'yellow');
    step(state.videoPlayer, im_show);
end

end

function labels = gaussian_shaped_labels(sigma, sz)
[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
labels = exp(-0.5 / sigma^2 * (rs.^2 + cs.^2));
labels = circshift(labels, -floor(sz(1:2) / 2) + 1);
assert(labels(1,1) == 1)
end

function img_crop = imcrop_multiscale(img, pos, sz, output_sz, yyxx)
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

g = bsxfun(@times, yyxx, s); % scale
g = bsxfun(@plus, g, t); % translate
g = reshape(g, 2, output_sz(1), output_sz(2), []);

img_crop = vl_nnbilinearsampler(img, g);
end
