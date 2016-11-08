function net = cnn_hog_init(varargin)
rng('default');
rng(0) ;

net = dagnn.DagNN() ;
net.meta.normalization.averageImage = reshape(single([123.6800,116.7790 ,103.9390]),[1,1,3]);
net.meta.normalization.imageSize = [256,256,3];

conv1 = dagnn.Conv('size', [3 3 3 64], 'pad', 1, 'stride', 1, 'hasBias', true) ;
net.addLayer('conv1', conv1, {'image'}, {'conv1'}, {'filters1', 'biases1'}) ;
net.addLayer('relu1', dagnn.ReLU(), {'conv1'}, {'conv1x'});
pool1 = dagnn.Pooling('method', 'max', 'poolSize', [2 2],'pad', 0, 'stride', 2);
net.addLayer('pool1', pool1, {'conv1x'}, {'pool1'});

conv2 = dagnn.Conv('size', [3 3 64 64], 'pad', 1, 'stride', 1, 'hasBias', true) ;
net.addLayer('conv2', conv2, {'pool1'}, {'conv2'}, {'filters2', 'biases2'}) ;
net.addLayer('relu2', dagnn.ReLU(), {'conv2'}, {'conv2x'});
pool2 = dagnn.Pooling('method', 'max', 'poolSize', [2 2],'pad', 0, 'stride', 2);
net.addLayer('pool2', pool2, {'conv2x'}, {'pool2'});

conv3 = dagnn.Conv('size', [3 3 64 31], 'pad', 1, 'stride', 1, 'hasBias', true) ;
net.addLayer('conv3', conv3, {'pool2'}, {'conv3'}, {'filters3', 'biases3'}) ;
% net.addLayer('relu3', dagnn.ReLU(), {'conv3'}, {'conv3x'});

net.addLayer('lossl1', dagnn.LossL1(), {'conv3','hog'}, 'objective');

% Meta parameters
net.meta.inputSize = [256 256 3] ;

% Fill in defaul values
net.initParams();

end