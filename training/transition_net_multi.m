function transition_net_multi(varargin)

opts.expDir = fullfile('../data','DCFNet-net-7') ;
opts = vl_argparse(opts, varargin);

modelPath = dir(fullfile(opts.expDir, 'net-epoch-*.mat'));
modelPath = sort({modelPath.name});

[~,out_name,~] = fileparts(opts.expDir);

for i = 1:numel(modelPath)
    load(fullfile(opts.expDir, modelPath{i}));
    net = dagnn.DagNN.loadobj(net) ;
    net = deployDCFNet(net);
    if ~exist('../model_multi', 'dir')
        mkdir('../model_multi');
    end
    save(fullfile('../model_multi', [out_name num2str(i, 'epoch_%d.mat')]), 'net') ;
end
end
