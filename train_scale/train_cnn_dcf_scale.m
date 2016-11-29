function [net, info] = train_cnn_dcf_scale(varargin)
%CNN_DCF
run('vl_setupnn.m') ;
fftw('planner','patient');
opts.network = [] ;
opts.networkType = 'dagnn' ;
opts.expDir = fullfile('../data', 'vot-fc-dcf-scale') ;
opts.dataDir = fullfile('../data', 'vot16') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.lite = ismac();

if ispc()
    trainOpts.gpus = [2];
else
    trainOpts.gpus = [];
end
trainOpts.learningRate = 1e-5;
trainOpts.weightDecay = 0.0005;
trainOpts.numEpochs = 100;
trainOpts.batchSize = 1;
opts.train = trainOpts;

if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if isempty(opts.network)
  net = fc_dcf_scale_init() ;
else
  net = opts.network ;
  opts.network = [] ;
end

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

switch opts.networkType
  case 'simplenn', trainfn = @cnn_train ;
  case 'dagnn', trainfn = @cnn_train_dag ;
end

[net, info] = trainfn(net, imdb, getBatch(opts), ...
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
if opts.numGpus > 0
    target_cpu = vl_imreadjpeg(imdb.images.target(batch));
    target = gpuArray(target_cpu{1});
    bboxs_target = gpuArray(imdb.images.bboxs_target(batch,:));
    search_cpu = vl_imreadjpeg(imdb.images.target(batch));
    search = gpuArray(search_cpu{1});
    bboxs_search = gpuArray(imdb.images.bboxs_search(batch,:));
elses
    target_cpu = vl_imreadjpeg(imdb.images.target(batch));
    target = (target_cpu{1});
    bboxs_target = (imdb.images.bboxs_target(batch,:));
    
    search_cpu = vl_imreadjpeg(imdb.images.target(batch));
    search = (search_cpu{1});
    bboxs_search = (imdb.images.bboxs_search(batch,:));
end

inputs = {'target', target,'search',search,...
    'bboxs_target',bboxs_target,'bboxs_search',bboxs_search} ;
end