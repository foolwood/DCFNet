net = load('imagenet-caffe-alex.mat');
net = vl_simplenn_tidy(net) ;
net.layers(4:end) = [];
net.layers{1, 3}.precious = true;
net.layers{1, 1}.pad = [5,5,5,5];
net.layers{1, 1}.stride = [1,1];

im = imread('peppers.png') ;
im_ = single(im) ; % note: 255 range
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
im_ = im_ - net.meta.normalization.averageImage ;
res = vl_simplenn(net, im_) ;

scores = squeeze(gather(res(end).x)) ;

net = load('imagenet-vgg-verydeep-16.mat');
net = vl_simplenn_tidy(net) ;
net.layers(5:end) = [];
net.layers{1, 4}.precious = true;