function [net, info] = train_cnn_dcf(varargin)
%CNN_DCF
run('vl_setupnn.m') ;
fftw('planner','patient');
opts.networkType = 1;
opts.lossType = 1;
opts.expDir = fullfile('../data',...
    ['vot2016-' num2str(opts.networkType) '-' num2str(opts.lossType) '-DCFNet']) ;
opts.dataDir = fullfile('../data', 'vot16') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.lite = false;

trainOpts.learningRate = 1e-5;
trainOpts.momentum = 0.9;
trainOpts.weightDecay = 0.0005;
trainOpts.numEpochs = 50;
trainOpts.batchSize = 1;
trainOpts.prefetch = true ; 
opts.train = trainOpts;

if ~isfield(opts.train, 'gpus') && ispc(), opts.train.gpus = [1];
else opts.train.gpus = []; end

% --------------------------------------------------------------------
%                                                 Prepare net and data
% --------------------------------------------------------------------
net = cnn_dcf_init(opts.networkType, opts.lossType);

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getVOTImdb('lite',opts.lite) ;
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

netStruct = net.saveobj() ;
save('./vgg16_dcf.mat', '-v7.3', '-struct', 'netStruct') ;
clear netStruct ;

end

% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
bopts = struct('numGpus', numel(opts.train.gpus),'sz', [125,125]) ;
fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% --------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------
if nargout == 0
    vl_imreadjpeg([imdb.images.target(batch),imdb.images.search(batch)],'prefetch') ;
    inputs = []; 
    return;
end
    
if opts.numGpus > 0
    target_gpu = vl_imreadjpeg(imdb.images.target(batch),'GPU');
    target = target_gpu{1};
    search_gpu = vl_imreadjpeg(imdb.images.search(batch),'GPU');
    search = search_gpu{1};
    bbox_target = gpuArray(imdb.images.target_bboxs(batch,:));
    bbox_search = gpuArray(imdb.images.search_bboxs(batch,:));
else
    target_cpu = vl_imreadjpeg(imdb.images.target(batch));
    target = target_cpu{1};
    search_cpu = vl_imreadjpeg(imdb.images.search(batch));
    search = search_cpu{1};
    bbox_target = imdb.images.target_bboxs(batch,:);
    bbox_search = imdb.images.search_bboxs(batch,:);
end

% target_cpu = vl_imreadjpeg(imdb.images.target(1),'NumThreads',32);
% target = target_cpu{1};
% search = target;
% bbox_target = imdb.images.target_bboxs(batch,:);
% bbox_search = bbox_target;

% bbox2rect = @(x) ([x(1)+1,x(2)+1,x(3)-x(1),x(4)-x(2)]);
% subplot(1,2,1);imshow(uint8(target));rectangle('Position',bbox2rect(bbox_target));
% subplot(1,2,2);imshow(uint8(search));rectangle('Position',bbox2rect(bbox_search));

inputs = {'bbox_target',bbox_target,'bbox_search',bbox_search,...
    'image_target', target, 'image_search', search} ;
end