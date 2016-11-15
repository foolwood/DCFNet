function net = cnn_hog_l2_init(varargin)
rng('default');
rng(0) ;

net = dagnn.DagNN() ;
net.meta.normalization.averageImage = reshape(single([123.6800,116.7790 ,103.9390]),[1,1,3]);
net.meta.normalization.imageSize = [256,256,3];

conv1_1 = dagnn.Conv('size', [3 3 3 64], 'pad', 1, 'stride', 1, 'hasBias', true) ;
net.addLayer('conv1_1', conv1_1, {'image'}, {'conv1_1'}, {'conv1_1f', 'conv1_1b'}) ;
net.addLayer('relu1_1', dagnn.ReLU(), {'conv1_1'}, {'conv1_1x'});

conv1_2 = dagnn.Conv('size', [3 3 64 64], 'pad', 1, 'stride', 1, 'hasBias', true) ;
net.addLayer('conv1_2', conv1_2, {'conv1_1x'}, {'conv1_2'}, {'conv1_2f', 'conv1_2b'}) ;
net.addLayer('relu1_2', dagnn.ReLU(), {'conv1_2'}, {'conv1_2x'});

pool1 = dagnn.Pooling('method', 'max', 'poolSize', [2 2],'pad', 0, 'stride', 2);
net.addLayer('pool1', pool1, {'conv1_2x'}, {'pool1'});

conv2_1 = dagnn.Conv('size', [3 3 64 128], 'pad', 1, 'stride', 1, 'hasBias', true) ;
net.addLayer('conv2_1', conv2_1, {'pool1'}, {'conv2_1'}, {'conv2_1f', 'conv2_1b'}) ;
net.addLayer('relu2_1', dagnn.ReLU(), {'conv2_1'}, {'conv2_1x'});

conv2_2 = dagnn.Conv('size', [3 3 128 128], 'pad', 1, 'stride', 1, 'hasBias', true) ;
net.addLayer('conv2_2', conv2_2, {'conv2_1x'}, {'conv2_2'}, {'conv2_2f', 'conv2_2b'}) ;
net.addLayer('relu2_2', dagnn.ReLU(), {'conv2_2'}, {'conv2_2x'});

pool2 = dagnn.Pooling('method', 'max', 'poolSize', [2 2],'pad', 0, 'stride', 2);
net.addLayer('pool2', pool2, {'conv2_2x'}, {'pool2'});

conv3_1 = dagnn.Conv('size', [3 3 128 256], 'pad', 1, 'stride', 1, 'hasBias', true) ;
net.addLayer('conv3_1', conv3_1, {'pool2'}, {'conv3_1'}, {'conv3_1f', 'conv3_1b'}) ;
net.addLayer('relu3_1', dagnn.ReLU(), {'conv3_1'}, {'conv3_1x'});

conv3_2 = dagnn.Conv('size', [3 3 256 256], 'pad', 1, 'stride', 1, 'hasBias', true) ;
net.addLayer('conv3_2', conv3_2, {'conv3_1x'}, {'conv3_2'}, {'conv3_2f', 'conv3_2b'}) ;
net.addLayer('relu3_2', dagnn.ReLU(), {'conv3_2'}, {'conv3_2x'});

conv3_3 = dagnn.Conv('size', [3 3 256 31], 'pad', 1, 'stride', 1, 'hasBias', true) ;
net.addLayer('conv3_3', conv3_3, {'conv3_2x'}, {'conv3_3'}, {'conv3_3f', 'conv3_3b'}) ;
net.addLayer('relu3_3', dagnn.ReLU(), {'conv3_3'}, {'conv3_3x'});

net.addLayer('lossl2', dagnn.LossL2(), {'conv3_3x','hog'}, 'objective');

% Meta parameters
net.meta.inputSize = [256 256 3] ;
% Fill in defaul values
net.initParams();

end