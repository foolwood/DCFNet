function net = cnn_l1_init(varargin)
rng('default');
rng(0) ;

net = dagnn.DagNN() ;
net.meta.normalization.averageImage = reshape(single([123.6800,116.7790 ,103.9390]),[1,1,3]);
net.meta.normalization.imageSize = [256,256,3];


conv1 = dagnn.Conv('size', [1 1 3 1], 'pad', 0, 'stride', 1, 'hasBias', true) ;
net.addLayer('conv1', conv1, {'image_rgb'}, {'conv1'}, {'conv1f', 'conv1b'}) ;

net.addLayer('lossl1', dagnn.LossL1(), {'conv1','image_gray'}, 'objective');

% Meta parameters
net.meta.inputSize = [256 256 3] ;
% Fill in defaul values
net.initParams();

end