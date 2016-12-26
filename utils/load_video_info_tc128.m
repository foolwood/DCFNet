function [img_files, ground_truth] = load_video_info_tc128(base_path, video)

video_path = fullfile(base_path,video);

filename = fullfile(video_path,[video '_gt.txt']);
ground_truth = dlmread(filename);
ground_truth = [ground_truth(:,1),ground_truth(:,2),...
    ground_truth(:,1),ground_truth(:,2)+ground_truth(:,4),...
    ground_truth(:,1)+ground_truth(:,3),ground_truth(:,2)+ground_truth(:,4),...
    ground_truth(:,1)+ground_truth(:,3),ground_truth(:,2)];

filename = fullfile(video_path,[video '_frames.txt']);
numframe = csvread(filename);
img_files = num2str((numframe(1) : numframe(2))', '%04i.jpg');
img_files = cellstr(img_files);
img_files = fullfile(video_path,'img',img_files);
end

