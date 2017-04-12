function transition_net(varargin)
%CNN_DCF
run('vl_setupnn.m') ;
opts.dataset = 3;
opts.networkType = 5;
opts.lossType = 1;
opts.expDir = fullfile('../data',...
    ['dataset-',num2str(opts.dataset),'-net-',num2str(opts.networkType),'-loss-' num2str(opts.lossType) '-DCFNet-New']) ;


modelPath = dir(fullfile(opts.expDir, 'net-epoch-*0.mat'));
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
