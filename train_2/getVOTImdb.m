function imdb = getVOTImdb(varargin)
rng('default');
addpath('../utils');
opts = [];
opts.dataDir = '../data';
opts.vot16_dataDir = fullfile(opts.dataDir,'VOT16');
opts.visualization = false;
opts.lite = false;
opts = vl_argparse(opts, varargin);
imdb = [];
bbox_mode = 'axis_aligned';
net_input_size = [125,125,3];

lite_index = true(21395,1);
if opts.lite
    lite_index = false(21395,1);
    lite_index(randperm(21395,100)) = true;
end

% -------------------------------------------------------------------------
%           vot16:21395
% -------------------------------------------------------------------------
set = [ones(1,21395)];

if strcmp(bbox_mode,'axis_aligned'),
    get_bbox = @get_axis_aligned_BB;
else
    get_bbox = @get_minmax_BB;
end

imdb.images.set = set;
imdb.images.set(randperm(21395,100)) = 2;
imdb.images.target = zeros([net_input_size,numel(set)],'uint8');
imdb.images.search = zeros([net_input_size,numel(set)],'uint8');
imdb.images.delta_yx = zeros(numel(set),2,'int16');

now_index = 1;
if opts.visualization
    subplot(1,2,1),f_1 = imshow(zeros(net_input_size));title('target');
    subplot(1,2,2),f_2 = imshow(zeros(net_input_size));hold on ;
    f_3 = plot(net_input_size(1)/2,net_input_size(2)/2,'r*');
    title('search');
    drawnow;
end
% -------------------------------------------------------------------------
%                                                                     VOT16
% -------------------------------------------------------------------------

disp('VOT2016 Data:');
vot16_dataDir = opts.vot16_dataDir;
videos = importdata(fullfile(vot16_dataDir,'list.txt'));

for v = 1:numel(videos)
    video = videos{v};fprintf('%3d :%20s\n',v,video);
    [img_files, ground_truth_4xy] = load_video_info_vot(vot16_dataDir, video);
    bbox_gt = get_bbox(ground_truth_4xy);
    
    pos_gt = round([(bbox_gt(:,2)+bbox_gt(:,4)),(bbox_gt(:,1)+bbox_gt(:,3))]/2)+1;
    sz_gt = round([(bbox_gt(:,4)-bbox_gt(:,2)),(bbox_gt(:,3)-bbox_gt(:,1))]);
    
    for frame = 1:(numel(img_files)-1)
        if lite_index(now_index)
            pos = pos_gt(frame,:);
            win_sz = sz_gt(frame,:)*(1+1.5);
            delta_yx_raw = pos_gt(frame+1,:)-pos_gt(frame,:);
            
            im_prev = imread(img_files{frame});
            im_curr = imread(img_files{frame+1});
            
            imdb.images.target(:,:,:,now_index) = imresize(...
                get_subwindow(im_prev, pos, win_sz),net_input_size(1:2));
            imdb.images.search(:,:,:,now_index) = imresize(...
                get_subwindow(im_curr, pos, win_sz),net_input_size(1:2));
            
            imdb.images.delta_yx(now_index,1:2) = delta_yx_raw.*net_input_size(1:2)./win_sz;
            
            if opts.visualization
                f_1.set('CData',imdb.images.target(:,:,:,now_index));
                f_2.set('CData',imdb.images.search(:,:,:,now_index));
                f_3.set('XData',imdb.images.delta_yx(now_index,2)+net_input_size(2)/2,...
                    'YData',imdb.images.delta_yx(now_index,1)+net_input_size(1)/2);
                drawnow;
            end
        end
        now_index = now_index+1;
    end %%end frame
end %%end v


if opts.lite
    imdb.images.target = imdb.images.target(:,:,:,lite_index);
    imdb.images.search = imdb.images.search(:,:,:,lite_index);
    imdb.images.set = imdb.images.set(lite_index);
    imdb.images.delta_yx = imdb.images.delta_yx(lite_index,:);
end
dataMean(1,1,1:3) = single([123,117,104]);
imdb.images.data_mean(1,1,1:3) = dataMean;
imdb.meta.sets = {'train', 'val'} ;
end %%end function



