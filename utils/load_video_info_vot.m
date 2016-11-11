function [img_files, ground_truth] = load_video_info_vot(base_path, video)

video_path = fullfile(base_path,video);

filename = fullfile(video_path,'groundtruth.txt');
ground_truth = csvread(filename);
if(size(ground_truth,2) == 4)%vot13
    ground_truth = [ground_truth(:,1),ground_truth(:,2),...
        ground_truth(:,1),ground_truth(:,2)+ground_truth(:,4),...
        ground_truth(:,1)+ground_truth(:,3),ground_truth(:,2)+ground_truth(:,4),...
        ground_truth(:,1)+ground_truth(:,3),ground_truth(:,2)];
end

img_files = dir(fullfile(video_path,'*.jpg'));
if isempty(img_files),
    error('No image files to load.');
end
img_files = sort({img_files.name});
img_files = fullfile(video_path,img_files);
end

