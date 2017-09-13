function [net, info] = train_DCFNet(varargin)
%Training DCFNet
training_env();
opts.networkType = 12;
opts.lossType = 1;
opts.inputSize = 125;
opts.padding = 2; % padding size for crop
opts.gpus = 1;
[opts] = vl_argparse(opts, varargin) ;

opts.expDir = fullfile('..', 'data', sprintf('DCFNet-net-%d-%d-%1.1f', opts.networkType, opts.inputSize, opts.padding)) ;
opts.imdbPath = fullfile('..', 'data', sprintf('imdb-vid2015-DCFNet-%d-%1.1f', opts.inputSize, opts.padding), 'imdb.mat');

trainOpts.momentum = 0.9;
trainOpts.weightDecay = 0.0005;
trainOpts.numEpochs = 50;
trainOpts.learningRate = logspace(-2, -5, trainOpts.numEpochs) ; % from SiameseFC
trainOpts.batchSize = 32;
trainOpts.gpus = [opts.gpus]; %only support single gpu
opts.train = trainOpts;

if ~isfield(opts.train, 'gpus')
    opts.train.gpus = [];
elseif numel(opts.train.gpus) ~=0
    gpuDevice(opts.train.gpus);
end

% --------------------------------------------------------------------
%                                                 Prepare net and data
% --------------------------------------------------------------------
net = init_DCFNet('networkType', opts.networkType,...
    'inputSize', opts.inputSize, 'padding', opts.padding);

if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath) ;
else
    imdb = getImdbDCFNet('output_size', opts.inputSize,...
        'padding', opts.padding) ;
    if ~exist(fileparts(opts.imdbPath), 'dir')
        mkdir(fileparts(opts.imdbPath));
    end
    save(opts.imdbPath, '-v7.3', '-struct', 'imdb') ;
end

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
[net, info] = cnn_train_dag(net, imdb, getBatch(opts), ...
    'expDir', opts.expDir, opts.train, 'val', find(imdb.images.set == 2)) ;

transition_net('expDir', opts.expDir);
transition_net_multi('expDir', opts.expDir);
end

% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
bopts = struct('numGpus', numel(opts.train.gpus), 'batchSize', opts.train.batchSize, 'maxStep', 10) ;
fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% --------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------
rand_next = randi([1, opts.maxStep], size(batch));
rand_next = min(rand_next, imdb.images.up_index(batch));

if opts.numGpus > 0
    target = gpuArray(single(imdb.images.images(:,:,:,batch)));
    search = gpuArray(single(imdb.images.images(:,:,:,batch+rand_next)));
else
    target = single(imdb.images.images(:,:,:,batch));
    search = single(imdb.images.images(:,:,:,batch+rand_next));
end
target = bsxfun(@minus, target, imdb.images.data_mean);
search = bsxfun(@minus, search, imdb.images.data_mean);
inputs = {'target', target, 'search', search} ;
end