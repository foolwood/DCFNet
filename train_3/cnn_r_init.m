function net = cnn_r_init(varargin)
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
conv1_1 = dagnn.Conv('size', [3 3 3 64], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
net.addLayer('conv1_1', conv1_1, {'target'}, {'conv1_1'}, {'conv1_1f', 'conv1_1b'}) ;
net.addLayer('relu1_1', dagnn.ReLU(), {'conv1_1'}, {'conv1_1x'});

conv1_2 = dagnn.Conv('size', [3 3 64 64], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
net.addLayer('conv1_2', conv1_2, {'conv1_1x'}, {'conv1_2'}, {'conv1_2f', 'conv1_2b'}) ;
net.addLayer('relu1_2', dagnn.ReLU(), {'conv1_2'}, {'conv1_2x'});

conv2_1 = dagnn.Conv('size', [3 3 64 128], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
net.addLayer('conv2_1', conv2_1, {'conv1_2x'}, {'conv2_1'}, {'conv2_1f', 'conv2_1b'}) ;
net.addLayer('relu2_1', dagnn.ReLU(), {'conv2_1'}, {'conv2_1x'});

conv2_2 = dagnn.Conv('size', [3 3 128 128], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
net.addLayer('conv2_2', conv2_2, {'conv2_1x'}, {'conv2_2'}, {'conv2_2f', 'conv2_2b'}) ;
net.addLayer('relu2_2', dagnn.ReLU(), {'conv2_2'}, {'conv2_2x'});

conv3_1 = dagnn.Conv('size', [3 3 128 256], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
net.addLayer('conv3_1', conv3_1, {'conv2_2x'}, {'conv3_1'}, {'conv3_1f', 'conv3_1b'}) ;
net.addLayer('relu3_1', dagnn.ReLU(), {'conv3_1'}, {'conv3_1x'});

conv3_2 = dagnn.Conv('size', [3 3 256 256], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
net.addLayer('conv3_2', conv3_2, {'conv3_1x'}, {'conv3_2'}, {'conv3_2f', 'conv3_2b'}) ;
net.addLayer('relu3_2', dagnn.ReLU(), {'conv3_2'}, {'conv3_2x'});

conv3_3 = dagnn.Conv('size', [3 3 256 256], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
net.addLayer('conv3_3', conv3_3, {'conv3_2x'}, {'conv3_3'}, {'conv3_3f', 'conv3_3b'}) ;
net.addLayer('relu3_3', dagnn.ReLU(), {'conv3_3'}, {'conv3_3x'});

%% search
conv1_1s = dagnn.Conv('size', [3 3 3 64], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
net.addLayer('conv1_1s', conv1_1s, {'search'}, {'conv1_1s'}, {'conv1_1f', 'conv1_1b'}) ;
net.addLayer('relu1_1s', dagnn.ReLU(), {'conv1_1s'}, {'conv1_1sx'});

conv1_2s = dagnn.Conv('size', [3 3 64 64], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
net.addLayer('conv1_2s', conv1_2s, {'conv1_1sx'}, {'conv1_2s'}, {'conv1_2f', 'conv1_2b'}) ;
net.addLayer('relu1_2s', dagnn.ReLU(), {'conv1_2s'}, {'conv1_2sx'});

conv2_1s = dagnn.Conv('size', [3 3 64 128], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
net.addLayer('conv2_1s', conv2_1s, {'conv1_2sx'}, {'conv2_1s'}, {'conv2_1f', 'conv2_1b'}) ;
net.addLayer('relu2_1s', dagnn.ReLU(), {'conv2_1s'}, {'conv2_1sx'});

conv2_2s = dagnn.Conv('size', [3 3 128 128], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
net.addLayer('conv2_2s', conv2_2s, {'conv2_1sx'}, {'conv2_2s'}, {'conv2_2f', 'conv2_2b'}) ;
net.addLayer('relu2_2s', dagnn.ReLU(), {'conv2_2s'}, {'conv2_2sx'});

conv3_1s = dagnn.Conv('size', [3 3 128 256], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
net.addLayer('conv3_1s', conv3_1s, {'conv2_2sx'}, {'conv3_1s'}, {'conv3_1f', 'conv3_1b'}) ;
net.addLayer('relu3_1s', dagnn.ReLU(), {'conv3_1s'}, {'conv3_1sx'});

conv3_2s = dagnn.Conv('size', [3 3 256 256], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
net.addLayer('conv3_2s', conv3_2s, {'conv3_1sx'}, {'conv3_2s'}, {'conv3_2f', 'conv3_2b'}) ;
net.addLayer('relu3_2s', dagnn.ReLU(), {'conv3_2s'}, {'conv3_2sx'});

conv3_3s = dagnn.Conv('size', [3 3 256 256], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
net.addLayer('conv3_3s', conv3_3s, {'conv3_2sx'}, {'conv3_3s'}, {'conv3_3f', 'conv3_3b'}) ;
net.addLayer('relu3_3s', dagnn.ReLU(), {'conv3_3s'}, {'conv3_3sx'});


%% Concat

net.addLayer('concat' , dagnn.Concat(), {'conv3_3x','conv3_3sx'}, {'conv3_concat'}) ;

conv4_1 = dagnn.Conv('size', [3 3 512 512], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
net.addLayer('conv4_1', conv4_1, {'conv3_concat'}, {'conv4_1'}, {'conv4_1f', 'conv4_1b'}) ;
net.addLayer('relu4_1', dagnn.ReLU(), {'conv4_1'}, {'conv4_1x'});

conv4_2 = dagnn.Conv('size', [3 3 512 512], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
net.addLayer('conv4_2', conv4_2, {'conv4_1x'}, {'conv4_2'}, {'conv4_2f', 'conv4_2b'}) ;
net.addLayer('relu4_2', dagnn.ReLU(), {'conv4_2'}, {'conv4_2x'});

conv4_3 = dagnn.Conv('size', [3 3 512 1], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
net.addLayer('conv4_3', conv4_3, {'conv4_2x'}, {'response'}, {'conv4_3f', 'conv4_3b'}) ;

window_sz = [125,125];
target_sz = [50,50];
sigma = sqrt(prod(target_sz))/10;

CResponseLossL1 = dagnn.CResponseLossL1('win_size', window_sz,'sigma',sigma) ;
net.addLayer('CResponseLossL1', CResponseLossL1, {'response','delta_yx'}, 'objective') ;

CCenterLoss = dagnn.CCenterLoss('win_size', window_sz) ;
net.addLayer('CCenterLoss', CCenterLoss, {'response','delta_yx'}, 'CLE') ;

% Fill in defaul values
net.initParams();

%% meta
net.meta.normalization.imageSize = [125,125,3];
net.meta.normalization.averageImage = reshape(single([123,117,104]),[1,1,3]);

%% Save

netStruct = net.saveobj() ;
save('../model/cnn_dcf.mat', '-v7.3', '-struct', 'netStruct') ;
clear netStruct ;

end
