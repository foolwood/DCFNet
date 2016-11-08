function [net, info] = cnn_hog(varargin)
%CNN_HOG  Demonstrates CNN Expression
run('vl_setupnn.m') ;

opts.network = [] ;
opts.feature = 'hog';
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
trainOpts.learningRate = 1e-5;
trainOpts.weightDecay = 0.0005;
trainOpts.numEpochs = 50;
trainOpts.batchSize = 50;
opts.train = trainOpts;

opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if isempty(opts.network)
  net = cnn_hog_init('networkType', opts.networkType) ;
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
                      'val', imdb.images.set == 2) ;
end




% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
bopts = struct('numGpus', numel(opts.train.gpus),'sz',[227,227]) ;
fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% --------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------

if opts.numGpus > 0
    images = vl_imreadjpeg(imdb.images.datapath(batch),'NumThreads',5,...
        'Pack','CropLocation','random','Resize',[256 256],'SubtractAverage',imdb.images.data_mean) ;
    images = images{1};
    
    hog = zeros(64,64,32,size(images,4),'single');
    for i = 1:size(images,4)
        hog(1:64,1:64,1:32,i) = fhog(images(:,:,:,i), 4, 9);
    end
    hog(:,:,32,:) = [];
    images = gpuArray(images);
    hog = gpuArray(hog);
else
    images = vl_imreadjpeg(imdb.images.datapath(batch),'NumThreads',32,...
        'Pack','CropLocation','random','Resize',[256 256],'SubtractAverage',imdb.images.data_mean) ;
    images = images{1};
    hog = zeros(64,64,32,size(images,4),'single');
    
    for i = 1:size(images,4)
        hog(1:64,1:64,1:32,i) = fhog(images(:,:,:,i), 4, 9);
    end
    hog(:,:,32,:) = [];
end

inputs = {'image', images, 'hog', hog} ;
end

% --------------------------------------------------------------------
function imdb = getHogImdb(opts)
% --------------------------------------------------------------------
datapath = dir(fullfile(opts.dataDir,'*.jpg'));
datapath = fullfile(opts.dataDir,{datapath.name});

dataMean = [123.6800,116.7790 ,103.9390];
dataMean = reshape(dataMean,1,1,[]);

set = [ones(1,round(numel(datapath)*0.8)),...
    2*ones(1,round(numel(datapath)*0.2))];

imdb.images.datapath = datapath ;
imdb.images.data_mean = dataMean;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val'} ;
end
