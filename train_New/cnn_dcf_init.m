function net = cnn_dcf_init(networkType, lossType)
% input:
%       -target :125*125*3*n
%       -search :125*125*3*n
%       -delta_xy :n*2
% output:
%       -response :125*125*1*n(test)
rng('default');
rng(0) ;

net = dagnn.DagNN() ;

%% meta
net.meta.normalization.imageSize = [125,125,3];
net.meta.normalization.averageImage = reshape(single([123,117,104]),[1,1,3]);

if networkType == 1
    %% target
    conv1 = dagnn.Conv('size', [1 1 3 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1', conv1, {'target'}, {'conv1'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('norm1', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv1'}, {'x'});
    
    %% search
    conv1s = dagnn.Conv('size', [1 1 3 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1s', conv1s, {'search'}, {'conv1s'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('norm1s', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv1s'}, {'z'});
elseif networkType == 2
    %% target
    conv1 = dagnn.Conv('size', [1 1 3 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1', conv1, {'target'}, {'conv1'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1', dagnn.ReLU(), {'conv1'}, {'conv1x'});
    
    conv2 = dagnn.Conv('size', [1 1 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2', conv2, {'conv1x'}, {'conv2'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('norm1', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv2'}, {'x'});
    
    %% search
    conv1s = dagnn.Conv('size', [1 1 3 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1s', conv1s, {'search'}, {'conv1s'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1s', dagnn.ReLU(), {'conv1s'}, {'conv1sx'});
    
    conv2s = dagnn.Conv('size', [1 1 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2s', conv2s, {'conv1sx'}, {'conv2s'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('norm1s', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv2s'}, {'z'});
elseif networkType == 3
    %% target
    conv1 = dagnn.Conv('size', [1 1 3 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1', conv1, {'target'}, {'conv1'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1', dagnn.ReLU(), {'conv1'}, {'conv1x'});
    
    conv2 = dagnn.Conv('size', [1 1 3 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2', conv2, {'conv1x'}, {'conv2'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('relu2', dagnn.ReLU(), {'conv2'}, {'conv2x'});
    
    conv3 = dagnn.Conv('size', [1 1 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv3', conv3, {'conv2x'}, {'conv3'}, {'conv3f', 'conv3b'}) ;
    net.addLayer('norm1', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv3'}, {'x'});
    
    %% search
    conv1s = dagnn.Conv('size', [1 1 3 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1s', conv1s, {'search'}, {'conv1s'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1s', dagnn.ReLU(), {'conv1s'}, {'conv1sx'});
    
    conv2s = dagnn.Conv('size', [1 1 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2s', conv2s, {'conv1sx'}, {'conv2s'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('relu2s', dagnn.ReLU(), {'conv2s'}, {'conv2sx'});
    
    conv3s = dagnn.Conv('size', [1 1 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv3s', conv3s, {'conv2sx'}, {'conv3s'}, {'conv3f', 'conv3b'}) ;
    net.addLayer('norm1s', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv3s'}, {'z'});
end

%% dcf
window_sz = [125,125];
target_sz = [50,50];
sigma = sqrt(prod(target_sz))/10;
DCF = dagnn.DCF('win_size', window_sz,'sigma', sigma) ;
net.addLayer('DCF', DCF, {'x','z'}, {'response'}) ;

if lossType == 1
    ResponseLossL2 = dagnn.ResponseLossL2('win_size', window_sz,'sigma',sigma) ;
    net.addLayer('ResponseLoss', ResponseLossL2, {'response'}, 'objective') ;
else
    ResponseLossL2 = dagnn.ResponseLossL2('win_size', window_sz,'sigma',sigma) ;
    net.addLayer('ResponseLoss', ResponseLossL2, {'response'}, 'objective') ;
end

CenterLoss = dagnn.CenterLoss('win_size', window_sz) ;
net.addLayer('CenterLoss', CenterLoss, {'response'}, 'CLE') ;

% Fill in defaul values
net.initParams();
% vgg16_net = load('../model/imagenet-vgg-verydeep-16.mat') ;
% vgg16_net = dagnn.DagNN.fromSimpleNN(vgg16_net);
% net.params(net.getParamIndex('conv1_1f')) = vgg16_net.params(net.getParamIndex('conv1_1f'));
% net.params(net.getParamIndex('conv1_2f')) = vgg16_net.params(net.getParamIndex('conv1_2f'));
% net.params(net.getParamIndex('conv2_1f')) = vgg16_net.params(net.getParamIndex('conv2_1f'));
% net.params(net.getParamIndex('conv2_2f')) = vgg16_net.params(net.getParamIndex('conv2_2f'));
% net.params(net.getParamIndex('conv3_1f')) = vgg16_net.params(net.getParamIndex('conv3_1f'));
% net.params(net.getParamIndex('conv3_2f')) = vgg16_net.params(net.getParamIndex('conv3_2f'));
% net.params(net.getParamIndex('conv3_3f')) = vgg16_net.params(net.getParamIndex('conv3_3f'));
%
% net.params(net.getParamIndex('conv1_1b')) = vgg16_net.params(net.getParamIndex('conv1_1b'));
% net.params(net.getParamIndex('conv1_2b')) = vgg16_net.params(net.getParamIndex('conv1_2b'));
% net.params(net.getParamIndex('conv2_1b')) = vgg16_net.params(net.getParamIndex('conv2_1b'));
% net.params(net.getParamIndex('conv2_2b')) = vgg16_net.params(net.getParamIndex('conv2_2b'));
% net.params(net.getParamIndex('conv3_1b')) = vgg16_net.params(net.getParamIndex('conv3_1b'));
% net.params(net.getParamIndex('conv3_2b')) = vgg16_net.params(net.getParamIndex('conv3_2b'));
% net.params(net.getParamIndex('conv3_3b')) = vgg16_net.params(net.getParamIndex('conv3_3b'));

%% Save

% netStruct = net.saveobj() ;
% save('../model/cnn_dcf.mat', '-v7.3', '-struct', 'netStruct') ;
% clear netStruct ;

end
