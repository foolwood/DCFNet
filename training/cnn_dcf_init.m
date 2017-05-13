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
    window_sz = [125,125];
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
    window_sz = [125,125];
elseif networkType == 3
    %% target
    conv1 = dagnn.Conv('size', [1 1 3 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1', conv1, {'target'}, {'conv1'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1', dagnn.ReLU(), {'conv1'}, {'conv1x'});
    
    conv2 = dagnn.Conv('size', [1 1 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
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
    window_sz = [125,125];
elseif networkType == 4
    %% target
    conv1 = dagnn.Conv('size', [1 1 3 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1', conv1, {'target'}, {'conv1'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1', dagnn.ReLU(), {'conv1'}, {'conv1x'});
    
    conv2 = dagnn.Conv('size', [1 1 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2', conv2, {'conv1x'}, {'conv2'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('relu2', dagnn.ReLU(), {'conv2'}, {'conv2x'});
    
    conv3 = dagnn.Conv('size', [1 1 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv3', conv3, {'conv2x'}, {'conv3'}, {'conv3f', 'conv3b'}) ;
    net.addLayer('relu3', dagnn.ReLU(), {'conv3'}, {'conv3x'});
    
    conv4 = dagnn.Conv('size', [1 1 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv4', conv4, {'conv3x'}, {'conv4'}, {'conv4f', 'conv4b'}) ;
    net.addLayer('norm1', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv4'}, {'x'});
    
    %% search
    conv1s = dagnn.Conv('size', [1 1 3 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1s', conv1s, {'search'}, {'conv1s'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1s', dagnn.ReLU(), {'conv1s'}, {'conv1sx'});
    
    conv2s = dagnn.Conv('size', [1 1 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2s', conv2s, {'conv1sx'}, {'conv2s'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('relu2s', dagnn.ReLU(), {'conv2s'}, {'conv2sx'});

    conv3s = dagnn.Conv('size', [1 1 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv3s', conv3s, {'conv2sx'}, {'conv3s'}, {'conv3f', 'conv3b'}) ;
    net.addLayer('relu3s', dagnn.ReLU(), {'conv3s'}, {'conv3sx'});
    
    conv4s = dagnn.Conv('size', [1 1 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv4s', conv4s, {'conv3sx'}, {'conv4s'}, {'conv4f', 'conv4b'}) ;
    net.addLayer('norm1s', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv4s'}, {'z'});
    window_sz = [125,125];
elseif networkType == 5
    %% target
    conv1 = dagnn.Conv('size', [1 1 3 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1', conv1, {'target'}, {'conv1'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1', dagnn.ReLU(), {'conv1'}, {'conv1x'});
    
    conv2 = dagnn.Conv('size', [1 1 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2', conv2, {'conv1x'}, {'conv2'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('relu2', dagnn.ReLU(), {'conv2'}, {'conv2x'});
    
    conv3 = dagnn.Conv('size', [1 1 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv3', conv3, {'conv2x'}, {'conv3'}, {'conv3f', 'conv3b'}) ;
    net.addLayer('relu3', dagnn.ReLU(), {'conv3'}, {'conv3x'});

    conv4 = dagnn.Conv('size', [1 1 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv4', conv4, {'conv3x'}, {'conv4'}, {'conv4f', 'conv4b'}) ;
    net.addLayer('relu4', dagnn.ReLU(), {'conv4'}, {'conv4x'});
    
    conv5 = dagnn.Conv('size', [1 1 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv5', conv5, {'conv4x'}, {'conv5'}, {'conv5f', 'conv5b'}) ;
    net.addLayer('norm1', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv5'}, {'x'});
    
    %% search
    conv1s = dagnn.Conv('size', [1 1 3 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1s', conv1s, {'search'}, {'conv1s'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1s', dagnn.ReLU(), {'conv1s'}, {'conv1sx'});
    
    conv2s = dagnn.Conv('size', [1 1 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2s', conv2s, {'conv1sx'}, {'conv2s'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('relu2s', dagnn.ReLU(), {'conv2s'}, {'conv2sx'});

    conv3s = dagnn.Conv('size', [1 1 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv3s', conv3s, {'conv2sx'}, {'conv3s'}, {'conv3f', 'conv3b'}) ;
    net.addLayer('relu3s', dagnn.ReLU(), {'conv3s'}, {'conv3sx'});

    conv4s = dagnn.Conv('size', [1 1 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv4s', conv4s, {'conv3sx'}, {'conv4s'}, {'conv4f', 'conv4b'}) ;
    net.addLayer('relu4s', dagnn.ReLU(), {'conv4s'}, {'conv4sx'});
    
    conv5s = dagnn.Conv('size', [1 1 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv5s', conv5s, {'conv4sx'}, {'conv5s'}, {'conv5f', 'conv5b'}) ;
    net.addLayer('norm1s', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv5s'}, {'z'});
    window_sz = [125,125];
elseif networkType == 6
    %% target
    conv1 = dagnn.Conv('size', [3 3 3 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1', conv1, {'target'}, {'conv1'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('norm1', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv1'}, {'x'});
    
    %% search
    conv1s = dagnn.Conv('size', [3 3 3 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1s', conv1s, {'search'}, {'conv1s'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('norm1s', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv1s'}, {'z'});
    window_sz = [125,125];
elseif networkType == 7
    %% target
    conv1 = dagnn.Conv('size', [3 3 3 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1', conv1, {'target'}, {'conv1'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1', dagnn.ReLU(), {'conv1'}, {'conv1x'});
    
    conv2 = dagnn.Conv('size', [3 3 32 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2', conv2, {'conv1x'}, {'conv2'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('norm1', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv2'}, {'x'});
    
    %% search
    conv1s = dagnn.Conv('size', [3 3 3 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1s', conv1s, {'search'}, {'conv1s'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1s', dagnn.ReLU(), {'conv1s'}, {'conv1sx'});
    
    conv2s = dagnn.Conv('size', [3 3 32 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2s', conv2s, {'conv1sx'}, {'conv2s'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('norm1s', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv2s'}, {'z'});
    window_sz = [125,125];
elseif networkType == 8
    %% target
    conv1 = dagnn.Conv('size', [3 3 3 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1', conv1, {'target'}, {'conv1'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1', dagnn.ReLU(), {'conv1'}, {'conv1x'});
    
    conv2 = dagnn.Conv('size', [3 3 32 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2', conv2, {'conv1x'}, {'conv2'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('relu2', dagnn.ReLU(), {'conv2'}, {'conv2x'});
    
    conv3 = dagnn.Conv('size', [3 3 32 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv3', conv3, {'conv2x'}, {'conv3'}, {'conv3f', 'conv3b'}) ;
    net.addLayer('norm1', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv3'}, {'x'});
    
    %% search
    conv1s = dagnn.Conv('size', [3 3 3 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1s', conv1s, {'search'}, {'conv1s'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1s', dagnn.ReLU(), {'conv1s'}, {'conv1sx'});
    
    conv2s = dagnn.Conv('size', [3 3 32 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2s', conv2s, {'conv1sx'}, {'conv2s'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('relu2s', dagnn.ReLU(), {'conv2s'}, {'conv2sx'});
    
    conv3s = dagnn.Conv('size', [3 3 32 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv3s', conv3s, {'conv2sx'}, {'conv3s'}, {'conv3f', 'conv3b'}) ;
    net.addLayer('norm1s', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv3s'}, {'z'});
    window_sz = [125,125];
elseif networkType == 9
    %% target
    conv1 = dagnn.Conv('size', [3 3 3 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1', conv1, {'target'}, {'conv1'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1', dagnn.ReLU(), {'conv1'}, {'conv1x'});
    
    conv2 = dagnn.Conv('size', [3 3 32 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2', conv2, {'conv1x'}, {'conv2'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('relu2', dagnn.ReLU(), {'conv2'}, {'conv2x'});
    
    conv3 = dagnn.Conv('size', [3 3 32 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv3', conv3, {'conv2x'}, {'conv3'}, {'conv3f', 'conv3b'}) ;
    net.addLayer('relu3', dagnn.ReLU(), {'conv3'}, {'conv3x'});
    
    conv4 = dagnn.Conv('size', [3 3 32 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv4', conv4, {'conv3x'}, {'conv4'}, {'conv4f', 'conv4b'}) ;
    net.addLayer('norm1', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv4'}, {'x'});
    
    %% search
    conv1s = dagnn.Conv('size', [3 3 3 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1s', conv1s, {'search'}, {'conv1s'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1s', dagnn.ReLU(), {'conv1s'}, {'conv1sx'});
    
    conv2s = dagnn.Conv('size', [3 3 32 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2s', conv2s, {'conv1sx'}, {'conv2s'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('relu2s', dagnn.ReLU(), {'conv2s'}, {'conv2sx'});

    conv3s = dagnn.Conv('size', [3 3 32 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv3s', conv3s, {'conv2sx'}, {'conv3s'}, {'conv3f', 'conv3b'}) ;
    net.addLayer('relu3s', dagnn.ReLU(), {'conv3s'}, {'conv3sx'});
    
    conv4s = dagnn.Conv('size', [3 3 32 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv4s', conv4s, {'conv3sx'}, {'conv4s'}, {'conv4f', 'conv4b'}) ;
    net.addLayer('norm1s', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv4s'}, {'z'});
    window_sz = [125,125];
elseif networkType == 10
    %% target
    conv1 = dagnn.Conv('size', [3 3 3 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1', conv1, {'target'}, {'conv1'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1', dagnn.ReLU(), {'conv1'}, {'conv1x'});
    
    conv2 = dagnn.Conv('size', [3 3 32 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2', conv2, {'conv1x'}, {'conv2'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('relu2', dagnn.ReLU(), {'conv2'}, {'conv2x'});
    
    conv3 = dagnn.Conv('size', [3 3 32 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv3', conv3, {'conv2x'}, {'conv3'}, {'conv3f', 'conv3b'}) ;
    net.addLayer('relu3', dagnn.ReLU(), {'conv3'}, {'conv3x'});

    conv4 = dagnn.Conv('size', [3 3 32 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv4', conv4, {'conv3x'}, {'conv4'}, {'conv4f', 'conv4b'}) ;
    net.addLayer('relu4', dagnn.ReLU(), {'conv4'}, {'conv4x'});
    
    conv5 = dagnn.Conv('size', [3 3 32 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv5', conv5, {'conv4x'}, {'conv5'}, {'conv5f', 'conv5b'}) ;
    net.addLayer('norm1', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv5'}, {'x'});
    
    %% search
    conv1s = dagnn.Conv('size', [3 3 3 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1s', conv1s, {'search'}, {'conv1s'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1s', dagnn.ReLU(), {'conv1s'}, {'conv1sx'});
    
    conv2s = dagnn.Conv('size', [3 3 32 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2s', conv2s, {'conv1sx'}, {'conv2s'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('relu2s', dagnn.ReLU(), {'conv2s'}, {'conv2sx'});

    conv3s = dagnn.Conv('size', [3 3 32 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv3s', conv3s, {'conv2sx'}, {'conv3s'}, {'conv3f', 'conv3b'}) ;
    net.addLayer('relu3s', dagnn.ReLU(), {'conv3s'}, {'conv3sx'});

    conv4s = dagnn.Conv('size', [3 3 32 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv4s', conv4s, {'conv3sx'}, {'conv4s'}, {'conv4f', 'conv4b'}) ;
    net.addLayer('relu4s', dagnn.ReLU(), {'conv4s'}, {'conv4sx'});
    
    conv5s = dagnn.Conv('size', [3 3 32 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv5s', conv5s, {'conv4sx'}, {'conv5s'}, {'conv5f', 'conv5b'}) ;
    net.addLayer('norm1s', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv5s'}, {'z'});
    window_sz = [125,125];
elseif networkType == 11
    %% target
    conv1 = dagnn.Conv('size', [3 3 3 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1', conv1, {'target'}, {'conv1'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('norm1', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv1'}, {'x'});
    
    %% search
    conv1s = dagnn.Conv('size', [3 3 3 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1s', conv1s, {'search'}, {'conv1s'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('norm1s', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv1s'}, {'z'});
    window_sz = [123,123];
elseif networkType == 12
    %% target
    conv1 = dagnn.Conv('size', [3 3 3 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1', conv1, {'target'}, {'conv1'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1', dagnn.ReLU(), {'conv1'}, {'conv1x'});
    
    conv2 = dagnn.Conv('size', [3 3 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2', conv2, {'conv1x'}, {'conv2'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('norm1', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv2'}, {'x'});
    
    %% search
    conv1s = dagnn.Conv('size', [3 3 3 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1s', conv1s, {'search'}, {'conv1s'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1s', dagnn.ReLU(), {'conv1s'}, {'conv1sx'});
    
    conv2s = dagnn.Conv('size', [3 3 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2s', conv2s, {'conv1sx'}, {'conv2s'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('norm1s', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv2s'}, {'z'});
    window_sz = [121,121];
elseif networkType == 13
    %% target
    conv1 = dagnn.Conv('size', [3 3 3 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1', conv1, {'target'}, {'conv1'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1', dagnn.ReLU(), {'conv1'}, {'conv1x'});
    
    conv2 = dagnn.Conv('size', [3 3 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2', conv2, {'conv1x'}, {'conv2'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('relu2', dagnn.ReLU(), {'conv2'}, {'conv2x'});
    
    conv3 = dagnn.Conv('size', [3 3 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv3', conv3, {'conv2x'}, {'conv3'}, {'conv3f', 'conv3b'}) ;
    net.addLayer('norm1', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv3'}, {'x'});
    
    %% search
    conv1s = dagnn.Conv('size', [3 3 3 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1s', conv1s, {'search'}, {'conv1s'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1s', dagnn.ReLU(), {'conv1s'}, {'conv1sx'});
    
    conv2s = dagnn.Conv('size', [3 3 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2s', conv2s, {'conv1sx'}, {'conv2s'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('relu2s', dagnn.ReLU(), {'conv2s'}, {'conv2sx'});
    
    conv3s = dagnn.Conv('size', [3 3 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv3s', conv3s, {'conv2sx'}, {'conv3s'}, {'conv3f', 'conv3b'}) ;
    net.addLayer('norm1s', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv3s'}, {'z'});
    window_sz = [119,119];
elseif networkType == 14
    %% target
    conv1 = dagnn.Conv('size', [3 3 3 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1', conv1, {'target'}, {'conv1'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1', dagnn.ReLU(), {'conv1'}, {'conv1x'});
    
    conv2 = dagnn.Conv('size', [3 3 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2', conv2, {'conv1x'}, {'conv2'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('relu2', dagnn.ReLU(), {'conv2'}, {'conv2x'});
    
    conv3 = dagnn.Conv('size', [3 3 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv3', conv3, {'conv2x'}, {'conv3'}, {'conv3f', 'conv3b'}) ;
    net.addLayer('relu3', dagnn.ReLU(), {'conv3'}, {'conv3x'});
    
    conv4 = dagnn.Conv('size', [3 3 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv4', conv4, {'conv3x'}, {'conv4'}, {'conv4f', 'conv4b'}) ;
    net.addLayer('norm1', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv4'}, {'x'});
    
    %% search
    conv1s = dagnn.Conv('size', [3 3 3 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1s', conv1s, {'search'}, {'conv1s'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1s', dagnn.ReLU(), {'conv1s'}, {'conv1sx'});
    
    conv2s = dagnn.Conv('size', [3 3 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2s', conv2s, {'conv1sx'}, {'conv2s'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('relu2s', dagnn.ReLU(), {'conv2s'}, {'conv2sx'});

    conv3s = dagnn.Conv('size', [3 3 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv3s', conv3s, {'conv2sx'}, {'conv3s'}, {'conv3f', 'conv3b'}) ;
    net.addLayer('relu3s', dagnn.ReLU(), {'conv3s'}, {'conv3sx'});
    
    conv4s = dagnn.Conv('size', [3 3 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv4s', conv4s, {'conv3sx'}, {'conv4s'}, {'conv4f', 'conv4b'}) ;
    net.addLayer('norm1s', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv4s'}, {'z'});
    window_sz = [117,117];
elseif networkType == 15
    %% target
    conv1 = dagnn.Conv('size', [3 3 3 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1', conv1, {'target'}, {'conv1'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1', dagnn.ReLU(), {'conv1'}, {'conv1x'});
    
    conv2 = dagnn.Conv('size', [3 3 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2', conv2, {'conv1x'}, {'conv2'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('relu2', dagnn.ReLU(), {'conv2'}, {'conv2x'});
    
    conv3 = dagnn.Conv('size', [3 3 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv3', conv3, {'conv2x'}, {'conv3'}, {'conv3f', 'conv3b'}) ;
    net.addLayer('relu3', dagnn.ReLU(), {'conv3'}, {'conv3x'});

    conv4 = dagnn.Conv('size', [3 3 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv4', conv4, {'conv3x'}, {'conv4'}, {'conv4f', 'conv4b'}) ;
    net.addLayer('relu4', dagnn.ReLU(), {'conv4'}, {'conv4x'});
    
    conv5 = dagnn.Conv('size', [3 3 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv5', conv5, {'conv4x'}, {'conv5'}, {'conv5f', 'conv5b'}) ;
    net.addLayer('norm1', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv5'}, {'x'});
    
    %% search
    conv1s = dagnn.Conv('size', [3 3 3 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1s', conv1s, {'search'}, {'conv1s'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1s', dagnn.ReLU(), {'conv1s'}, {'conv1sx'});
    
    conv2s = dagnn.Conv('size', [3 3 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2s', conv2s, {'conv1sx'}, {'conv2s'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('relu2s', dagnn.ReLU(), {'conv2s'}, {'conv2sx'});

    conv3s = dagnn.Conv('size', [3 3 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv3s', conv3s, {'conv2sx'}, {'conv3s'}, {'conv3f', 'conv3b'}) ;
    net.addLayer('relu3s', dagnn.ReLU(), {'conv3s'}, {'conv3sx'});

    conv4s = dagnn.Conv('size', [3 3 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv4s', conv4s, {'conv3sx'}, {'conv4s'}, {'conv4f', 'conv4b'}) ;
    net.addLayer('relu4s', dagnn.ReLU(), {'conv4s'}, {'conv4sx'});
    
    conv5s = dagnn.Conv('size', [3 3 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv5s', conv5s, {'conv4sx'}, {'conv5s'}, {'conv5f', 'conv5b'}) ;
    net.addLayer('norm1s', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv5s'}, {'z'});
    window_sz = [115,115];
elseif networkType == 16
    %% target
    conv1 = dagnn.Conv('size', [3 3 3 64], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1', conv1, {'target'}, {'conv1'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1', dagnn.ReLU(), {'conv1'}, {'conv1x'});
    
    conv2 = dagnn.Conv('size', [3 3 64 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2', conv2, {'conv1x'}, {'conv2'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('norm1', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv2'}, {'x'});
    
    %% search
    conv1s = dagnn.Conv('size', [3 3 3 64], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1s', conv1s, {'search'}, {'conv1s'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1s', dagnn.ReLU(), {'conv1s'}, {'conv1sx'});
    
    conv2s = dagnn.Conv('size', [3 3 64 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2s', conv2s, {'conv1sx'}, {'conv2s'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('norm1s', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv2s'}, {'z'});
    window_sz = [125,125];
elseif networkType == 17
    %% target
    conv1 = dagnn.Conv('size', [3 3 3 96], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1', conv1, {'target'}, {'conv1'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1', dagnn.ReLU(), {'conv1'}, {'conv1x'});
    
    conv2 = dagnn.Conv('size', [3 3 96 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2', conv2, {'conv1x'}, {'conv2'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('norm1', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv2'}, {'x'});
    
    %% search
    conv1s = dagnn.Conv('size', [3 3 3 96], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1s', conv1s, {'search'}, {'conv1s'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1s', dagnn.ReLU(), {'conv1s'}, {'conv1sx'});
    
    conv2s = dagnn.Conv('size', [3 3 96 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2s', conv2s, {'conv1sx'}, {'conv2s'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('norm1s', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv2s'}, {'z'});
    window_sz = [125,125];
elseif networkType == 18
    %% target
    conv1 = dagnn.Conv('size', [3 3 3 128], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1', conv1, {'target'}, {'conv1'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1', dagnn.ReLU(), {'conv1'}, {'conv1x'});
    
    conv2 = dagnn.Conv('size', [3 3 128 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2', conv2, {'conv1x'}, {'conv2'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('norm1', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv2'}, {'x'});
    
    %% search
    conv1s = dagnn.Conv('size', [3 3 3 128], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1s', conv1s, {'search'}, {'conv1s'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1s', dagnn.ReLU(), {'conv1s'}, {'conv1sx'});
    
    conv2s = dagnn.Conv('size', [3 3 128 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2s', conv2s, {'conv1sx'}, {'conv2s'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('norm1s', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv2s'}, {'z'});
    window_sz = [125,125];
elseif networkType == 19
    %% target
    conv1 = dagnn.Conv('size', [3 3 3 160], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1', conv1, {'target'}, {'conv1'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1', dagnn.ReLU(), {'conv1'}, {'conv1x'});
    
    conv2 = dagnn.Conv('size', [3 3 160 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2', conv2, {'conv1x'}, {'conv2'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('norm1', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv2'}, {'x'});
    
    %% search
    conv1s = dagnn.Conv('size', [3 3 3 160], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1s', conv1s, {'search'}, {'conv1s'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1s', dagnn.ReLU(), {'conv1s'}, {'conv1sx'});
    
    conv2s = dagnn.Conv('size', [3 3 160 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2s', conv2s, {'conv1sx'}, {'conv2s'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('norm1s', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv2s'}, {'z'});
    window_sz = [125,125];
elseif networkType == 20
    %% target
    conv1 = dagnn.Conv('size', [3 3 3 192], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1', conv1, {'target'}, {'conv1'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1', dagnn.ReLU(), {'conv1'}, {'conv1x'});
    
    conv2 = dagnn.Conv('size', [3 3 192 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2', conv2, {'conv1x'}, {'conv2'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('norm1', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv2'}, {'x'});
    
    %% search
    conv1s = dagnn.Conv('size', [3 3 3 192], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1s', conv1s, {'search'}, {'conv1s'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1s', dagnn.ReLU(), {'conv1s'}, {'conv1sx'});
    
    conv2s = dagnn.Conv('size', [3 3 192 32], 'pad', 1, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2s', conv2s, {'conv1sx'}, {'conv2s'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('norm1s', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv2s'}, {'z'});
    window_sz = [125,125];
    
elseif networkType == 21
    %% target
    conv1 = dagnn.Conv('size', [3 3 3 64], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1', conv1, {'target'}, {'conv1'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1', dagnn.ReLU(), {'conv1'}, {'conv1x'});
    
    conv2 = dagnn.Conv('size', [3 3 64 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2', conv2, {'conv1x'}, {'conv2'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('norm1', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv2'}, {'x'});
    
    %% search
    conv1s = dagnn.Conv('size', [3 3 3 64], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1s', conv1s, {'search'}, {'conv1s'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1s', dagnn.ReLU(), {'conv1s'}, {'conv1sx'});
    
    conv2s = dagnn.Conv('size', [3 3 64 32],'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2s', conv2s, {'conv1sx'}, {'conv2s'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('norm1s', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv2s'}, {'z'});
    window_sz = [121,121];
elseif networkType == 22
    %% target
    conv1 = dagnn.Conv('size', [3 3 3 96],'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1', conv1, {'target'}, {'conv1'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1', dagnn.ReLU(), {'conv1'}, {'conv1x'});
    
    conv2 = dagnn.Conv('size', [3 3 96 32],'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2', conv2, {'conv1x'}, {'conv2'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('norm1', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv2'}, {'x'});
    
    %% search
    conv1s = dagnn.Conv('size', [3 3 3 96],'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1s', conv1s, {'search'}, {'conv1s'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1s', dagnn.ReLU(), {'conv1s'}, {'conv1sx'});
    
    conv2s = dagnn.Conv('size', [3 3 96 32],'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2s', conv2s, {'conv1sx'}, {'conv2s'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('norm1s', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv2s'}, {'z'});
    window_sz = [121,121];
elseif networkType == 23
    %% target
    conv1 = dagnn.Conv('size', [3 3 3 128],'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1', conv1, {'target'}, {'conv1'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1', dagnn.ReLU(), {'conv1'}, {'conv1x'});
    
    conv2 = dagnn.Conv('size', [3 3 128 32],'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2', conv2, {'conv1x'}, {'conv2'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('norm1', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv2'}, {'x'});
    
    %% search
    conv1s = dagnn.Conv('size', [3 3 3 128],'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1s', conv1s, {'search'}, {'conv1s'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1s', dagnn.ReLU(), {'conv1s'}, {'conv1sx'});
    
    conv2s = dagnn.Conv('size', [3 3 128 32],'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2s', conv2s, {'conv1sx'}, {'conv2s'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('norm1s', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv2s'}, {'z'});
    window_sz = [121,121];
elseif networkType == 24
    %% target
    conv1 = dagnn.Conv('size', [3 3 3 160],'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1', conv1, {'target'}, {'conv1'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1', dagnn.ReLU(), {'conv1'}, {'conv1x'});
    
    conv2 = dagnn.Conv('size', [3 3 160 32],'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2', conv2, {'conv1x'}, {'conv2'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('norm1', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv2'}, {'x'});
    
    %% search
    conv1s = dagnn.Conv('size', [3 3 3 160],'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1s', conv1s, {'search'}, {'conv1s'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1s', dagnn.ReLU(), {'conv1s'}, {'conv1sx'});
    
    conv2s = dagnn.Conv('size', [3 3 160 32],'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2s', conv2s, {'conv1sx'}, {'conv2s'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('norm1s', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv2s'}, {'z'});
    window_sz = [121,121];
elseif networkType == 25
    %% target
    conv1 = dagnn.Conv('size', [3 3 3 192],'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1', conv1, {'target'}, {'conv1'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1', dagnn.ReLU(), {'conv1'}, {'conv1x'});
    
    conv2 = dagnn.Conv('size', [3 3 192 32],'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2', conv2, {'conv1x'}, {'conv2'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('norm1', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv2'}, {'x'});
    
    %% search
    conv1s = dagnn.Conv('size', [3 3 3 192],'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1s', conv1s, {'search'}, {'conv1s'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1s', dagnn.ReLU(), {'conv1s'}, {'conv1sx'});
    
    conv2s = dagnn.Conv('size', [3 3 192 32],'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2s', conv2s, {'conv1sx'}, {'conv2s'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('norm1s', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv2s'}, {'z'});
    window_sz = [121,121];
end


%% dcf
% window_sz = [125,125];
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
