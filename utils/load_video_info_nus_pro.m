function [img_files, ground_truth] = load_video_info_nus_pro(base_path, video)

video_path = fullfile(base_path,video);

filename = fullfile(video_path,'groundtruth.txt');
ground_truth = dlmread(filename);
ground_truth = [ground_truth(:,1),ground_truth(:,2),...
    ground_truth(:,1),ground_truth(:,4),...
    ground_truth(:,3),ground_truth(:,4),...
    ground_truth(:,3),ground_truth(:,2)];

img_files = dir(fullfile(video_path,'*.jpg'));
if isempty(img_files),
    error('No image files to load.')
end
img_files = sort({img_files.name});
img_files = fullfile(video_path,img_files);

filename = fullfile(video_path,'datainfo.txt');
numframe = csvread(filename);
numframe = numframe(3);
img_files = img_files(1:numframe);
end

