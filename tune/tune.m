% gpuDevice(2)
clear
close all
clc

vl_setupnn();
load('OTB2013_OPE_subS.mat');
for net_index = 6:6
    for scale_penalty = 0.99:0.005:1
        for output_sigma_factor = 0.1:0.1
            for num_scale = 3:2:3
                for interp_factor1 = 0.008:0.002:0.02
                    for interp_factor2 = 0.008:0.002:interp_factor1
                        param = struct();
                        param.net_index =  net_index;
                        param.interp_factor1 =  interp_factor1;
                        param.interp_factor2 =  interp_factor2;
                        param.num_scale =  num_scale;
                        param.scale_penalty =  scale_penalty;
                        param.output_sigma_factor =  output_sigma_factor;
                        
                        file = ['.\results\net_index-',num2str(net_index)...
                            '-interp_factor1-' num2str(interp_factor1)...
                            '-interp_factor2-' num2str(interp_factor2)...
                            '-num_scale-' num2str(num_scale)...
                            '-scale_penalty-' num2str(scale_penalty)...
                            '-output_sigma_factor-' num2str(output_sigma_factor)];
                        display(file)
                        if exist(file, 'dir') && numel(dir([file '\*.mat'])) == numel(otb_dataset)
                            continue;
                        end
                        if ~exist(file, 'dir')
                            mkdir(file);
                        end
                        main_tunning(file, param, otb_dataset);
                    end
                end
            end
        end
    end
end
exit