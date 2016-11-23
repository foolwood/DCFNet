function deepdcf_vot

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
[handle, image, region] = vot('rect');

vl_setupnn();
% Initialize the tracker
[state, ~] = deepdcf_initialize(imread(image), region);

while true
    
    % **********************************
    % VOT: Get next frame
    % **********************************
    [handle, image] = handle.frame(handle);
    
    if isempty(image)
        break;
    end;
    
    % Perform a tracking step, obtain new region
    [state, region] = deepdcf_update(state, imread(image));
    
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

function [state, location] = deepdcf_initialize(I, region, varargin)
location = region;

state.net = dagnn.DagNN.loadobj(load('vgg16_dcf.mat'));
state.net.mode = 'test';

state.lambda = 1e-4;
state.output_sigma = sqrt(prod([50,50]))/10;
state.interp_factor = 0.01;

state.yf = fft2(gaussian_shaped_labels(state.output_sigma, state.net.meta.normalization.imageSize(1:2)));

state.pos = region(2,1)+region(4,3)/2;
state.target_sz = region(4,3);


state.window_sz = state.target_sz*2.5;
patch = get_subwindow(I, state.pos, state.window_sz);

target = bsxfun(@minus,...
    single(imresize(patch,state.net.meta.normalization.imageSize(1:2))),...
    net.meta.normalization.averageImage);
xf = fft2(state.net.eval('image',target));

kf = linear_correlation(xf, xf);
state.model_alphaf = yf ./ (kf + state.lambda);
state.model_xf = xf;

end

function [state, location] = deepdcf_update(state, I, varargin)

patch = get_subwindow(I, state.pos, state.window_sz);

search = bsxfun(@minus,...
    single(imresize(patch,state.net.meta.normalization.imageSize(1:2))),...
    net.meta.normalization.averageImage);
zf = fft2(state.net.eval('image',search));

kzf = linear_correlation(zf, state.model_xf);

response = real(ifft2(state.model_alphaf .* kzf));
[vert_delta, horiz_delta] = find(response == max(response(:)), 1);
if vert_delta > size(zf,1) / 2,  %wrap around to negative half-space of vertical axis
    vert_delta = vert_delta - size(zf,1);
end
if horiz_delta > size(zf,2) / 2,  %same for horizontal axis
    horiz_delta = horiz_delta - size(zf,2);
end
state.pos = state.pos + [vert_delta - 1, horiz_delta - 1]*...
    state.window_sz./state.net.meta.normalization.imageSize(1:2);

patch = get_subwindow(I, state.pos, state.window_sz);
target = bsxfun(@minus,...
    single(imresize(patch,state.net.meta.normalization.imageSize(1:2))),...
    net.meta.normalization.averageImage);

xf = fft2(state.net.eval('image',target));
kf = linear_correlation(xf, xf);
alphaf = yf ./ (kf + state.lambda);   %equation for fast training

state.model_alphaf = (1 - state.interp_factor) * model_alphaf + state.interp_factor * alphaf;
state.model_xf = (1 - state.interp_factor) * model_xf + state.interp_factor * xf;

box = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];

location = double(box);

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



function kf = linear_correlation(xf, yf)
kf = sum(xf .* conj(yf), 3) / numel(xf);
end



function labels = gaussian_shaped_labels(sigma, sz)
[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
labels = exp(-0.5 / sigma^2 * (rs.^2 + cs.^2));
labels = circshift(labels, -floor(sz(1:2) / 2) + 1);
assert(labels(1,1) == 1)
end

