function transition_net(varargin)

opts.expDir = fullfile('../data','DCFNet-net-7') ;
opts = vl_argparse(opts, varargin);

modelPath = dir(fullfile(opts.expDir, 'net-epoch-*0.mat'));
modelPath = sort({modelPath.name});

[~,out_name,~] = fileparts(opts.expDir);

for i = numel(modelPath)
    load(fullfile(opts.expDir, modelPath{i}));
    net = dagnn.DagNN.loadobj(net) ;
    net = deployDCFNet(net);
    if ~exist('../model', 'dir')
        mkdir('../model');
    end
    save(fullfile('../model', [out_name '.mat']), 'net') ;
end
end
