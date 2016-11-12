function net = cnn_dcf_init(varargin)
% input:
%       -target :125*125*3*n
%       -search :125*125*3*n
%       -delta_xy :n*2
% output:
%       -response :125*125*1*n(test)
rng('default');
rng(0) ;

net = dagnn.DagNN() ;
%% target
conv1_1 = dagnn.Conv('size', [3 3 3 64], 'pad', 1, 'stride', 1, 'hasBias', true) ;
net.addLayer('conv1_1', conv1_1, {'target'}, {'conv1_1'}, {'conv1_1f', 'conv1_1b'}) ;
net.addLayer('relu1_1', dagnn.ReLU(), {'conv1_1'}, {'conv1_1x'});

conv1_2 = dagnn.Conv('size', [3 3 64 64], 'pad', 1, 'stride', 1, 'hasBias', true) ;
net.addLayer('conv1_2', conv1_2, {'conv1_1x'}, {'conv1_2'}, {'conv1_2f', 'conv1_2b'}) ;
net.addLayer('relu1_2', dagnn.ReLU(), {'conv1_2'}, {'conv1_2x'});

conv2_1 = dagnn.Conv('size', [3 3 64 128], 'pad', 1, 'stride', 1, 'hasBias', true) ;
net.addLayer('conv2_1', conv2_1, {'conv1_2x'}, {'conv2_1'}, {'conv2_1f', 'conv2_1b'}) ;
net.addLayer('relu2_1', dagnn.ReLU(), {'conv2_1'}, {'conv2_1x'});

conv2_2 = dagnn.Conv('size', [3 3 128 128], 'pad', 1, 'stride', 1, 'hasBias', true) ;
net.addLayer('conv2_2', conv2_2, {'conv2_1x'}, {'conv2_2'}, {'conv2_2f', 'conv2_2b'}) ;
net.addLayer('relu2_2', dagnn.ReLU(), {'conv2_2'}, {'conv2_2x'});

conv3_1 = dagnn.Conv('size', [3 3 128 256], 'pad', 1, 'stride', 1, 'hasBias', true) ;
net.addLayer('conv3_1', conv3_1, {'conv2_2x'}, {'conv3_1'}, {'conv3_1f', 'conv3_1b'}) ;
net.addLayer('relu3_1', dagnn.ReLU(), {'conv3_1'}, {'conv3_1x'});

conv3_2 = dagnn.Conv('size', [3 3 256 256], 'pad', 1, 'stride', 1, 'hasBias', true) ;
net.addLayer('conv3_2', conv3_2, {'conv3_1x'}, {'conv3_2'}, {'conv3_2f', 'conv3_2b'}) ;
net.addLayer('relu3_2', dagnn.ReLU(), {'conv3_2'}, {'conv3_2x'});

conv3_3 = dagnn.Conv('size', [3 3 256 256], 'pad', 1, 'stride', 1, 'hasBias', true) ;
net.addLayer('conv3_3', conv3_3, {'conv3_2x'}, {'conv3_3'}, {'conv3_3f', 'conv3_3b'}) ;
net.addLayer('relu3_3', dagnn.ReLU(), {'conv3_3'}, {'conv3_3x'});

%% search
conv1_1s = dagnn.Conv('size', [3 3 3 64], 'pad', 1, 'stride', 1, 'hasBias', true) ;
net.addLayer('conv1_1s', conv1_1s, {'search'}, {'conv1_1s'}, {'conv1_1f', 'conv1_1b'}) ;
net.addLayer('relu1_1s', dagnn.ReLU(), {'conv1_1s'}, {'conv1_1sx'});

conv1_2s = dagnn.Conv('size', [3 3 64 64], 'pad', 1, 'stride', 1, 'hasBias', true) ;
net.addLayer('conv1_2s', conv1_2s, {'conv1_1sx'}, {'conv1_2s'}, {'conv1_2f', 'conv1_2b'}) ;
net.addLayer('relu1_2s', dagnn.ReLU(), {'conv1_2s'}, {'conv1_2sx'});

conv2_1s = dagnn.Conv('size', [3 3 64 128], 'pad', 1, 'stride', 1, 'hasBias', true) ;
net.addLayer('conv2_1s', conv2_1s, {'conv1_2sx'}, {'conv2_1s'}, {'conv2_1f', 'conv2_1b'}) ;
net.addLayer('relu2_1s', dagnn.ReLU(), {'conv2_1s'}, {'conv2_1sx'});

conv2_2s = dagnn.Conv('size', [3 3 128 128], 'pad', 1, 'stride', 1, 'hasBias', true) ;
net.addLayer('conv2_2s', conv2_2s, {'conv2_1sx'}, {'conv2_2s'}, {'conv2_2f', 'conv2_2b'}) ;
net.addLayer('relu2_2s', dagnn.ReLU(), {'conv2_2s'}, {'conv2_2sx'});

conv3_1s = dagnn.Conv('size', [3 3 128 256], 'pad', 1, 'stride', 1, 'hasBias', true) ;
net.addLayer('conv3_1s', conv3_1s, {'conv2_2sx'}, {'conv3_1s'}, {'conv3_1f', 'conv3_1b'}) ;
net.addLayer('relu3_1s', dagnn.ReLU(), {'conv3_1s'}, {'conv3_1sx'});

conv3_2s = dagnn.Conv('size', [3 3 256 256], 'pad', 1, 'stride', 1, 'hasBias', true) ;
net.addLayer('conv3_2s', conv3_2s, {'conv3_1sx'}, {'conv3_2s'}, {'conv3_2f', 'conv3_2b'}) ;
net.addLayer('relu3_2s', dagnn.ReLU(), {'conv3_2s'}, {'conv3_2sx'});

conv3_3s = dagnn.Conv('size', [3 3 256 256], 'pad', 1, 'stride', 1, 'hasBias', true) ;
net.addLayer('conv3_3s', conv3_3s, {'conv3_2sx'}, {'conv3_3s'}, {'conv3_3f', 'conv3_3b'}) ;
net.addLayer('relu3_3s', dagnn.ReLU(), {'conv3_3s'}, {'conv3_3sx'});


%% dcf 
window_sz = [125,125];
target_sz = [50,50];
sigma = sqrt(prod(target_sz))/10;
DCF = dagnn.DCF('win_size', window_sz,'sigma',sigma) ;
net.addLayer('DCF', DCF, {'conv3_3x','conv3_3sx'}, {'response'}) ;

% ResponseLossSmoothL1 = dagnn.ResponseLossSmoothL1('win_size', window_sz,'sigma',sigma) ;
% net.addLayer('ResponseLoss', ResponseLossSmoothL1, {'response','delta_yx'}, 'objective') ;

ResponseLossL1 = dagnn.ResponseLossL1('win_size', window_sz,'sigma',sigma) ;
net.addLayer('ResponseLoss', ResponseLossL1, {'response','delta_yx'}, 'objective') ;

CenterLoss = dagnn.CenterLoss('win_size', window_sz) ;
net.addLayer('CenterLoss', CenterLoss, {'response','delta_yx'}, 'CLE') ;

% Fill in defaul values
% net.initParams();
vgg16_net = load('../model/imagenet-vgg-verydeep-16.mat') ;
vgg16_net = dagnn.DagNN.fromSimpleNN(vgg16_net);
net.params(net.getParamIndex('conv1_1f')) = vgg16_net.params(net.getParamIndex('conv1_1f'));
net.params(net.getParamIndex('conv1_2f')) = vgg16_net.params(net.getParamIndex('conv1_2f'));
net.params(net.getParamIndex('conv2_1f')) = vgg16_net.params(net.getParamIndex('conv2_1f'));
net.params(net.getParamIndex('conv2_2f')) = vgg16_net.params(net.getParamIndex('conv2_2f'));
net.params(net.getParamIndex('conv3_1f')) = vgg16_net.params(net.getParamIndex('conv3_1f'));
net.params(net.getParamIndex('conv3_2f')) = vgg16_net.params(net.getParamIndex('conv3_2f'));
net.params(net.getParamIndex('conv3_3f')) = vgg16_net.params(net.getParamIndex('conv3_3f'));

net.params(net.getParamIndex('conv1_1b')) = vgg16_net.params(net.getParamIndex('conv1_1b'));
net.params(net.getParamIndex('conv1_2b')) = vgg16_net.params(net.getParamIndex('conv1_2b'));
net.params(net.getParamIndex('conv2_1b')) = vgg16_net.params(net.getParamIndex('conv2_1b'));
net.params(net.getParamIndex('conv2_2b')) = vgg16_net.params(net.getParamIndex('conv2_2b'));
net.params(net.getParamIndex('conv3_1b')) = vgg16_net.params(net.getParamIndex('conv3_1b'));
net.params(net.getParamIndex('conv3_2b')) = vgg16_net.params(net.getParamIndex('conv3_2b'));
net.params(net.getParamIndex('conv3_3b')) = vgg16_net.params(net.getParamIndex('conv3_3b'));


%% meta
net.meta.normalization.imageSize = [125,125,3];
net.meta.normalization.averageImage = reshape(single([123,117,104]),[1,1,3]);

%% Save

netStruct = net.saveobj() ;
save('../model/cnn_dcf.mat', '-v7.3', '-struct', 'netStruct') ;
clear netStruct ;

end
