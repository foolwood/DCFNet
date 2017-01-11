function visual_boundingbox()
addpath('../utils');
result_path = 'D:\1\tracker_benchmark_v1.0\results';
load('OTB2013_OPE_subS.mat');

tracker_file = 'net_index-6-interp_factor-0.008-numScale-3-scale_penalty-0.99-output_sigma_factor-0.1';
for s = 1:numel(otb_dataset)
    matfile = fullfile(result_path,tracker_file,[otb_dataset(s).name,'_DCFNet.mat']);
    load(matfile);
    res = results{1,1}.res;
    update_visual_h = show_video(otb_dataset(s).s_frames, '', false);
    for frame = 1:numel(otb_dataset(s).s_frames)
        stop = update_visual_h(frame, res(frame,:));
        if stop,break;
        end
    end 
end