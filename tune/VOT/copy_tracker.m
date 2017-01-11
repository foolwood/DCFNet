report = fileread('tracker_DCFNet.m');

net_index = 6;
interp_factor = 0.01;
numScale = 3;
output_sigma_factor = 0.1;
scale_penalty = 0.97;
i = 0;
for net_index = 6:6
    for scale_penalty = 0.97:0.05:1
        for output_sigma_factor = 0.1:0.1
            for numScale = 3:5
                for interp_factor = 0:0.002:0.02
                    i = i+1;
                    filename = ['tracker_DCFNet',num2str(i),'.m'];
                    fid = fopen(filename, 'w');
                    modifiedStr = strrep(report, 'net_index = 6', ['net_index = ',num2str(net_index)]);
                    modifiedStr = strrep(modifiedStr, 'interp_factor = 0.01', ['interp_factor = ',num2str(interp_factor)]);
                    modifiedStr = strrep(modifiedStr, 'numScale = 3', ['numScale = ',num2str(numScale)]);
                    modifiedStr = strrep(modifiedStr, 'output_sigma_factor = 0.1', ['output_sigma_factor = ',num2str(output_sigma_factor)]);
                    modifiedStr = strrep(modifiedStr, 'scale_penalty = 0.97', ['scale_penalty = ',num2str(scale_penalty)]);
                    fwrite(fid, modifiedStr);
                    fclose(fid);
                end
            end
        end
    end
end
report = fileread('run_experiments.m');
for j = i:-1:1
    filename = ['run_experiments',num2str(j),'.m'];
    fid = fopen(filename, 'w');
    modifiedStr = strrep(report, 'DCFNet', ['DCFNet',num2str(j)]);
    fwrite(fid, modifiedStr);
    fclose(fid);
end

