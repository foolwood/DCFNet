function test_cnn_dcf()

image_file = dir('./Fish/img/*.jpg');
image_file = sort({image_file.name});
image_file = fullfile('./Fish/img/',image_file);


gt = dlmread('./Fish/groundtruth_rect.txt');
start_frame = 100;
next_frame = 10;

target_sz = gt(start_frame,[4,3]);
pos = gt(start_frame,[2,1])+floor(target_sz/2);
window_sz = floor(target_sz * (1 + 1.5));
sigma = sqrt(prod(target_sz))/10;

target_sz_2 = gt(next_frame,[4,3]);
pos_2 = gt(next_frame,[2,1])+floor(target_sz_2/2);
label_shift = pos_2 - pos;


im = imread(image_file{start_frame});
x = get_subwindow(im, pos, window_sz);
subplot(2,3,1),imshow(repmat(x,[1,1,3]));hold on;
plot(window_sz(2)/2,window_sz(1)/2,'r*');title('target');


im = imread(image_file{next_frame});
z = get_subwindow(im, pos, window_sz);
subplot(2,3,2),imshow(repmat(z,[1,1,3]));hold on;
plot(window_sz(2)/2+label_shift(2),window_sz(1)/2+label_shift(1),'r*');title('search');


netStruct = load('../model/cnn_dcf.mat') ;
% netStruct = load('./vgg_dcf.mat') ;
net = dagnn.DagNN.loadobj(netStruct) ;
clear netStruct ;
x = imresize(x,net.meta.normalization.imageSize(1:2));
x = repmat(x,[1,1,3]);
z = imresize(z,net.meta.normalization.imageSize(1:2));
z = repmat(z,[1,1,3]);

x = bsxfun(@minus,single(x),net.meta.normalization.averageImage);
z = bsxfun(@minus,single(z),net.meta.normalization.averageImage);


net.eval({'target',x,'search',z});
response = net.vars(net.getVarIndex('response')).value ;
response = imresize(response,window_sz);
subplot(2,3,3),imagesc(response);title('predic response');


[vert_delta, horiz_delta] = find(response == max(response(:)), 1);
if vert_delta > size(response,1) / 2,  %wrap around to negative half-space of vertical axis
    vert_delta = vert_delta - size(response,1);
end
if horiz_delta > size(response,2) / 2,  %same for horizontal axis
    horiz_delta = horiz_delta - size(response,2);
end

predic_pos = pos + [vert_delta - 1, horiz_delta - 1];
predic_rect = [predic_pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];

subplot(2,3,4),imshow(repmat(im,[1,1,3]));
rectangle('Position',predic_rect,'EdgeColor',[0,1,1]);
rectangle('Position',gt(next_frame,:),'EdgeColor',[0,1,0]);
title('predic rect');

% sz = [100,100];
% sigma = sqrt(prod(sz/2.5))/10;
% label_shift = [0,0];
% response = gaussian_shaped_labels_shift(sigma, sz, label_shift);
% subplot(2,3,5);imagesc(response);
net.eval({'target',x,'search',x});
response = net.vars(net.getVarIndex('response')).value ;
response = imresize(response,window_sz);
subplot(2,3,5);imagesc(response);title('learnt response');


response = gaussian_shaped_labels_shift(sigma, window_sz, label_shift);
subplot(2,3,6);imagesc(response);title('idea predict response');

end


function labels = gaussian_shaped_labels_shift(sigma, sz,label_shift)

[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
labels = exp(-0.5 / sigma^2 * (rs.^2 + cs.^2));

labels = circshift(labels, -floor(sz(1:2) / 2) + 1+label_shift);

% assert(labels(1,1) == 1)

end