function [net, info] = train_cnn_dcf(varargin)
%CNN_DCF
run('vl_setupnn.m') ;

opts.network = [] ;
opts.networkType = 'dagnn' ;
opts.expDir = fullfile('../data', 'vot-vgg-dcf') ;
opts.dataDir = fullfile('../data', 'vot16') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train = struct() ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if isempty(opts.network)
  net = cnn_dcf_init() ;
else
  net = opts.network ;
  opts.network = [] ;
end

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getVOTImdb(opts) ;
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
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 3)) ;
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
    target = vl_imreadjpeg(imdb.images.target(batch),...
        'NumThreads',32,'Pack','Resize',opts.sz,'SubtractAverage', imdb.images.data_mean,'GPU');
    search = vl_imreadjpeg(imdb.images.image(batch),...
        'NumThreads',32,'Pack','Resize',opts.sz,'SubtractAverage', imdb.images.data_mean,'GPU');
    delta_xy = gpuArray(single(imdb.images.bboxs(batch,1:2)));
else
    target = vl_imreadjpeg(imdb.images.target(batch),...
        'NumThreads',32,'Pack','Resize',opts.sz,'SubtractAverage', imdb.images.data_mean);
    search = vl_imreadjpeg(imdb.images.image(batch),...
        'NumThreads',32,'Pack','Resize',opts.sz,'SubtractAverage', imdb.images.data_mean);
    delta_xy = single(imdb.images.bboxs(batch,1:2));
end

inputs = {'target', target, 'search', search,'delta_xy',delta_xy} ;
end