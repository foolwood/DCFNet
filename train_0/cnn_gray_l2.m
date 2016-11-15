function [net, info] = cnn_l2(varargin)
%CNN_l2  Demonstrates L2 loss
run('vl_setupnn.m') ;

opts.network = [] ;
opts.feature = 'gray-l2';
opts.networkType = 'dagnn' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = [opts.networkType, opts.feature];
opts.expDir = fullfile('../data', ['DeepKCF-baseline-' sfx]) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile('../data', 'coco','train2014') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');


opts.train = struct() ;
if ispc()
    trainOpts.gpus = [1];
else
    trainOpts.gpus = [];
end
trainOpts.learningRate = 1e-6;
trainOpts.weightDecay = 0.0005;
trainOpts.numEpochs = 50;
trainOpts.batchSize = 1;
opts.train = trainOpts;

opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if isempty(opts.network)
  net = cnn_l2_init('networkType', opts.networkType) ;
else
  net = opts.network ;
  opts.network = [] ;
end

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getHogImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

switch opts.networkType
  case 'simplenn', trainfn = @cnn_train ;
  case 'dagnn', trainfn = @cnn_train_dag ;
end

[net, info] = trainfn(net, imdb, getBatch(opts), ...
                      'expDir', opts.expDir, ...
                      opts.train, ...
                      'val', find(imdb.images.set == 2)) ;
end




% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
conv_factor = reshape(single([0.3,0.3,0.3]),1,1,[]);
bopts = struct('numGpus', numel(opts.train.gpus),'conv_factor',conv_factor) ;
fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% --------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------

if opts.numGpus > 0
    image_rgb = vl_imreadjpeg(imdb.images.datapath(batch),'NumThreads',5,...
        'Pack','CropLocation','random','Resize',[256 256]) ;
    image_rgb = image_rgb{1};
    
    image_gray = sum(bsxfun(@times,image_rgb,opts.conv_factor),3);
   
    image_rgb = gpuArray(image_rgb);
    image_gray = gpuArray(image_gray);
else
    image_rgb = vl_imreadjpeg(imdb.images.datapath(batch),'NumThreads',5,...
        'Pack','CropLocation','random','Resize',[256 256]) ;
    image_rgb = image_rgb{1};
    
    image_gray = sum(bsxfun(@times,image_rgb,opts.conv_factor),3);
   
end

inputs = {'image_rgb', image_rgb, 'image_gray', image_gray} ;
end

% --------------------------------------------------------------------
function imdb = getHogImdb(opts)
% --------------------------------------------------------------------
datapath = dir(fullfile(opts.dataDir,'*.jpg'));
datapath = fullfile(opts.dataDir,{datapath.name});

dataMean = [123.6800,116.7790 ,103.9390];
dataMean = reshape(dataMean,1,1,[]);

set = [ones(1,round(numel(datapath)*0.9)),...
    2*ones(1,numel(datapath)-round(numel(datapath)*0.9))];

imdb.images.datapath = datapath ;
imdb.images.data_mean = dataMean;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val'} ;
end
