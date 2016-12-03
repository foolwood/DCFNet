function [net, info] = cnn_hog_l2(varargin)
%CNN_HOG  Demonstrates CNN Expression
run('vl_setupnn.m') ;
opts.expDir = fullfile('../data', 'CNN-HOG-L2') ;

opts.dataDir = fullfile('../data', 'coco','train2014') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');


opts.train = struct() ;
if ispc()
    trainOpts.gpus = [2];
else
    trainOpts.gpus = [];
end
trainOpts.learningRate = 1e-5;
trainOpts.weightDecay = 0.0005;
trainOpts.numEpochs = 50;
trainOpts.batchSize = 20;
opts.train = trainOpts;

opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

net = cnn_hog_l2_init() ;

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getHogImdb(opts) ;
  if ~exist(opts.expDir,'dir'),mkdir(opts.expDir) ;end
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

[net, info] = cnn_train_dag(net, imdb, getBatch(opts), ...
                      'expDir', opts.expDir, ...
                      opts.train, ...
                      'val', find(imdb.images.set == 2)) ;
end




% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
bopts = struct('numGpus', numel(opts.train.gpus)) ;
fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% --------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------

if opts.numGpus > 0
    images = vl_imreadjpeg(imdb.images.datapath(batch),'NumThreads',8,...
        'Pack','CropLocation','random','Resize',[256,256]) ;
    images = images{1};
    hog = zeros(64,64,32,size(images,4),'single');
    for i = 1:size(images,4)
        hog(1:64,1:64,1:32,i) = fhog(images(:,:,:,i), 4, 9);
    end
    hog(:,:,32,:) = [];
    images = gpuArray(images);
    hog = gpuArray(hog);
else
    images = vl_imreadjpeg(imdb.images.datapath(batch),'NumThreads',8,...
        'Pack','CropLocation','random','Resize',[256 256]) ;
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

set = [ones(1,round(numel(datapath)*0.9)),...
    2*ones(1,numel(datapath)-round(numel(datapath)*0.9))];

imdb.images.datapath = datapath ;
imdb.images.data_mean = dataMean;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val'} ;
end
