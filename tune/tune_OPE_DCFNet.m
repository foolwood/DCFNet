% gpuDevice(1)
net_name = dir('../model/*.mat');
interp_f = 0.00:0.002:0.03;
for net_index = 1:numel(net_name)
    for interp_factor = interp_f
        param.net_name = net_name(net_index).name;
        param.interp_factor = interp_factor;
        file = ['.\results\',param.net_name,...
            '-interp_factor-' num2str(interp_factor),'-OPE'];
        display(file)
        if exist(file, 'dir')
            continue;
        end
        if ~exist(file, 'dir')
            mkdir(file);
        end
        main_running_DCFNet(file,param,'OPE');
    end
end
exit