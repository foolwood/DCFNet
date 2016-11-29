function net = cnn_dcf_scale_init(varargin)
% input:
%         -target
%         -search
%         -bboxs_target
%         -bboxs_search
% output:
%       -response :n*33
rng('default');
rng(0) ;

net = dagnn.DagNN() ;

%% target

%% search

%% scale dcf 

%% Fill in defaul values
net.initParams();

%% meta

%% Save
% netStruct = net.saveobj() ;
% save('../model/cnn_dcf_scale.mat', '-v7.3', '-struct', 'netStruct') ;
% clear netStruct ;

end
