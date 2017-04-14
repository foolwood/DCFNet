function [net, info] = trainDCFNet(varargin)
%Training DCFNet
addpath('../utils/')
run('vl_setupnn.m') ;
fftw('planner','patient');
opts.dataset = 3;
opts.networkType = 21;
opts.lossType = 1;
opts.expDir = fullfile('../data',...
    ['dataset-',num2str(opts.dataset),'-net-',num2str(opts.networkType),'-loss-' num2str(opts.lossType) '-DCFNet-New']) ;
opts.imdbPath = fullfile('../data',['dataset-',num2str(opts.dataset)], 'imdb.mat');
opts.lite = false;

trainOpts.learningRate = 1e-5;
trainOpts.momentum = 0.9;
trainOpts.weightDecay = 0.0005;
trainOpts.numEpochs = 20;
trainOpts.batchSize = 16;
trainOpts.gpus = [1]; %only support single gpu
opts.train = trainOpts;

if ~isfield(opts.train, 'gpus')
    opts.train.gpus = [];
elseif numel(opts.train.gpus) ~=0
    gpuDevice(opts.train.gpus);
end

% --------------------------------------------------------------------
%                                                 Prepare net and data
% --------------------------------------------------------------------
net = cnn_dcf_init(opts.networkType, opts.lossType);

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getImdbRAM('dataset',opts.dataset) ;
  if ~exist(opts.expDir,'dir'),mkdir(opts.expDir);end
  save(opts.imdbPath, '-v7.3', '-struct', 'imdb') ;
end

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

[net, info] = cnn_train_dag(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  opts.train, ...
  'val', find(imdb.images.set == 2)) ;
% net = deployDCFNet(net);
% save(fullfile('../model',...
%     ['DCFNet-dataset-',num2str(opts.dataset),'-net-',num2str(opts.networkType),...
%     '-loss-' num2str(opts.lossType) '-epoch-' num2str(numel(info.train)) '.mat']),'net') ;

modelPath = dir(fullfile(opts.expDir, 'net-epoch-*.mat'));
modelPath = sort({modelPath.name});
for i = 1:numel(modelPath)
    load(fullfile(opts.expDir, modelPath{i}));
    net = dagnn.DagNN.loadobj(net) ;
    net = deployDCFNet(net);
    save(fullfile('../model',...
    ['DCFNet-dataset-',num2str(opts.dataset),'-net-',num2str(opts.networkType),...
    '-loss-' num2str(opts.lossType) modelPath{i}(4:end)]),'net') ;
end
end

% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
bopts = struct('numGpus', numel(opts.train.gpus),'batchSize',opts.train.batchSize,'step',10) ;
fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% --------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------
rand_next = randi([1,opts.step],size(batch));
rand_next = min(rand_next,imdb.images.up_index(batch));

if opts.numGpus > 0
    target = gpuArray(single(imdb.images.images(:,:,:,batch)));
    search = gpuArray(single(imdb.images.images(:,:,:,batch+rand_next)));
else
    target = single(imdb.images.images(:,:,:,batch));
    search = single(imdb.images.images(:,:,:,batch+rand_next));
end
% subplot(1,2,1),imshow(uint8(target));
% subplot(1,2,2),imshow(uint8(search));
% drawnow;
target = bsxfun(@minus,target,imdb.images.data_mean);
search = bsxfun(@minus,search,imdb.images.data_mean);
inputs = {'target',target,'search',search} ;
end