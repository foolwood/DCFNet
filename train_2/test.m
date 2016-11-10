
normalize = @(x,y)  ((double(x)/255) - mean(double(x(:)))/255).*y  ;


image_file = dir('./Fish/img/*.jpg');
image_file = sort({image_file.name});
image_file = fullfile('./Fish/img/',image_file);


gt = dlmread('./Fish/groundtruth_rect.txt');

target_sz = gt(1,[4,3]);
pos = gt(1,[2,1])+floor(target_sz/2);
window_sz = floor(target_sz * (1 + 1.5));
cos_window = single(hann(window_sz(1))) * single(hann(window_sz(2)))';	

im = imread(image_file{1});
x = get_subwindow(im, pos, window_sz);
subplot(2,2,1),imshow(repmat(x,[1,1,3]));
x = normalize(x,cos_window);

im = imread(image_file{24});
z = get_subwindow(im, pos, window_sz);
subplot(2,2,2),imshow(repmat(z,[1,1,3]));
z = normalize(z,cos_window);


net = dagnn.DagNN() ;
dcfBlock = dagnn.DCF('target_size', window_sz,'sigma',sqrt(prod(target_sz))/10) ;
net.addLayer('dcf', dcfBlock, {'x','z'}, {'response'}) ;

net.eval({'x',x,'z',z});

response = net.vars(net.getVarIndex('response')).value ;

subplot(2,2,3),imagesc(response);

[vert_delta, horiz_delta] = find(response == max(response(:)), 1);
if vert_delta > size(response,1) / 2,  %wrap around to negative half-space of vertical axis
    vert_delta = vert_delta - size(response,1);
end
if horiz_delta > size(response,2) / 2,  %same for horizontal axis
    horiz_delta = horiz_delta - size(response,2);
end

pos = pos + [vert_delta - 1, horiz_delta - 1];
predic_rect = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];

subplot(2,2,4),imshow(repmat(im,[1,1,3]));rectangle('Position',predic_rect);
