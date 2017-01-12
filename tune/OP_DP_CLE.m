ccc
addpath('D:\1\tracker_benchmark_v1.0\util');
addpath('D:\1\tracker_benchmark_v1.0\rstEval');

trackers = dir('D:\1\tracker_benchmark_v1.0\results\net_index-*');
valid_index = true(numel(trackers),1);
for i = 1:numel(trackers)
    matfile = dir(['D:\1\tracker_benchmark_v1.0\results\' trackers(i).name,'\*.mat']);
    if numel(matfile) < 51
        valid_index(i) = false;
        continue;
    end
    for j = 1:numel(matfile)
        load(['D:\1\tracker_benchmark_v1.0\results\' trackers(i).name,'\',matfile(j).name]);
        if isempty(results)
            valid_index(i) = false;
            continue;
        end
    end
end
trackers = trackers(valid_index);

seqs=configSeqs;
numSeg = 20;

num_seqs = numel(seqs);
num_trackers = numel(trackers);
OP_ALL = zeros(num_seqs,num_trackers);
DP_ALL = zeros(num_seqs,num_trackers);
CLE_ALL = zeros(num_seqs,num_trackers);
Speed_ALL = zeros(num_seqs,num_trackers);

for i = 1:num_seqs
    
    s = seqs{i};
    s.name
    s.len = s.endFrame - s.startFrame + 1;
    s.s_frames = cell(s.len,1);
    nz	= strcat('%0',num2str(s.nz),'d'); %number of zeros in the name of image
    for k=1:s.len
        image_no = s.startFrame + (k-1);
        id = sprintf(nz,image_no);
        s.s_frames{k} = strcat(s.path,id,'.',s.ext);
    end
   
    rect_anno = dlmread(['D:\1\tracker_benchmark_v1.0\anno\' s.name '.txt']);
    
    [subSeqs, subAnno]=splitSeqTRE(s,numSeg,rect_anno);
    ground_truth = subAnno{1};
    for j = 1:num_trackers
        matfile = fullfile('D:\1\tracker_benchmark_v1.0\results',...
            trackers(j).name,[s.name,'_DCFNet.mat']);
        load(matfile);
        
        [distance_precision, PASCAL_precision, average_center_location_error] = ...
            compute_performance_measures(results{1,1}.res, ground_truth);
        OP_ALL(i,j) = PASCAL_precision;
        DP_ALL(i,j) = distance_precision;
        CLE_ALL(i,j) = average_center_location_error;
        Speed_ALL(i,j) = results{1,1}.fps;
    end
end

median_OP = median(OP_ALL);
median_DP = median(DP_ALL);
median_CLE = median(CLE_ALL);
median_Speed = median(Speed_ALL);

figure(1)
plot(median_OP,'r');hold on;
plot(median_DP,'g');hold on;
plot(median_CLE,'b');hold on;


scale1_index = ~cell2mat(cellfun(@isempty,strfind({trackers.name},'numScale-1'),'UniformOutput',false));
scale3_index = ~cell2mat(cellfun(@isempty,strfind({trackers.name},'numScale-3'),'UniformOutput',false));
scale5_index = ~cell2mat(cellfun(@isempty,strfind({trackers.name},'numScale-5'),'UniformOutput',false));
scale7_index = ~cell2mat(cellfun(@isempty,strfind({trackers.name},'numScale-7'),'UniformOutput',false));

net6_index = ~cell2mat(cellfun(@isempty,strfind({trackers.name},'net_index-6'),'UniformOutput',false));
net7_index = ~cell2mat(cellfun(@isempty,strfind({trackers.name},'net_index-7'),'UniformOutput',false));
net8_index = ~cell2mat(cellfun(@isempty,strfind({trackers.name},'net_index-8'),'UniformOutput',false));

median_OP_temple = median_OP(net6_index&scale3_index);
median_DP_temple = median_DP(net6_index&scale3_index);
median_CLE_temple = median_CLE(net6_index&scale3_index);
median_Speed_temple = median_Speed(net6_index&scale3_index);
[~,median_OP_index] = max(median_OP_temple);
fprintf('DCFNet-conv1 & %2.2f & %2.2f & %2.2f & %2.2f \\\\\n',...
    median_OP_temple(median_OP_index),median_DP_temple(median_OP_index),...
    median_CLE_temple(median_OP_index),median_Speed_temple(median_OP_index)); 

median_OP_temple = median_OP(net7_index&scale3_index);
median_DP_temple = median_DP(net7_index&scale3_index);
median_CLE_temple = median_CLE(net7_index&scale3_index);
median_Speed_temple = median_Speed(net7_index&scale3_index);
[~,median_OP_index] = max(median_OP_temple);
fprintf('DCFNet-conv2 & %2.2f & %2.2f & %2.2f & %2.2f \\\\\n',...
    median_OP_temple(median_OP_index),median_DP_temple(median_OP_index),...
    median_CLE_temple(median_OP_index),median_Speed_temple(median_OP_index)); 

median_OP_temple = median_OP(net6_index&scale1_index);
median_DP_temple = median_DP(net6_index&scale1_index);
median_CLE_temple = median_CLE(net6_index&scale1_index);
median_Speed_temple = median_Speed(net6_index&scale1_index);
[~,median_OP_index] = max(0+median_OP_temple);
fprintf('DCFNet-conv1-1s & %2.2f & %2.2f & %2.2f & %2.2f \\\\\n',...
    median_OP_temple(median_OP_index),median_DP_temple(median_OP_index),...
    median_CLE_temple(median_OP_index),median_Speed_temple(median_OP_index)); 

median_OP_temple = median_OP(net6_index&scale5_index);
median_DP_temple = median_DP(net6_index&scale5_index);
median_CLE_temple = median_CLE(net6_index&scale5_index);
median_Speed_temple = median_Speed(net6_index&scale5_index);
[~,median_OP_index] = max(0+median_OP_temple);
fprintf('DCFNet-conv1-5s & %2.2f & %2.2f & %2.2f & %2.2f \\\\\n',...
    median_OP_temple(median_OP_index),median_DP_temple(median_OP_index),...
    median_CLE_temple(median_OP_index),median_Speed_temple(median_OP_index)); 

median_OP_temple = median_OP(net6_index&scale7_index);
median_DP_temple = median_DP(net6_index&scale7_index);
median_CLE_temple = median_CLE(net6_index&scale7_index);
median_Speed_temple = median_Speed(net6_index&scale7_index);
[~,median_OP_index] = max(0+median_OP_temple);
fprintf('DCFNet-conv1-7s & %2.2f & %2.2f & %2.2f & %2.2f \\\\\n',...
    median_OP_temple(median_OP_index),median_DP_temple(median_OP_index),...
    median_CLE_temple(median_OP_index),median_Speed_temple(median_OP_index)); 

median_OP_temple = median_OP(net8_index&scale3_index);
median_DP_temple = median_DP(net8_index&scale3_index);
median_CLE_temple = median_CLE(net8_index&scale3_index);
median_Speed_temple = median_Speed(net8_index&scale3_index);
[~,median_OP_index] = max(0+median_DP_temple);
fprintf('DCFNet-conv1-dilate & %2.2f & %2.2f & %2.2f & %2.2f \\\\\n',...
    median_OP_temple(median_OP_index),median_DP_temple(median_OP_index),...
    median_CLE_temple(median_OP_index),median_Speed_temple(median_OP_index)); 


