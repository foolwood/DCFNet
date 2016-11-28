function imdb = getVOTImdb(varargin)
addpath('../utils');
opts = [];
opts.dataDir = '../data';
opts.vot16_dataDir = fullfile(opts.dataDir,'VOT16');
opts.visualization = ismac();
opts.lite = ismac();
opts = vl_argparse(opts, varargin);
imdb = [];
bbox_mode = 'axis_aligned';

lite_index = true(21395,1);
if opts.lite
    lite_index = false(21395,1);
    lite_index(randperm(21395,1000)) = true;
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
imdb.images.target = cell([numel(set),1]);
imdb.images.search = cell([numel(set),1]);
imdb.images.target_bboxs = zeros(numel(set),4,'single');
imdb.images.search_bboxs = zeros(numel(set),4,'single');
now_index = 1;
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
    visualization = 1;
    for frame = 1:(numel(img_files)-1)
        if lite_index(now_index)
            
            imdb.images.target{now_index} = img_files{frame};
            imdb.images.search{now_index} = img_files{frame+1};
            imdb.images.target_bboxs(now_index,:) = bbox_gt(frame,:);
            imdb.images.search_bboxs(now_index,:) = bbox_gt(frame+1,:);
            
            
            if opts.visualization
                if visualization == 1
                    visualization = 2;
                    close all
                    subplot(1,2,1),f_1 = imshow(imread(imdb.images.target{now_index}));title('target');
                    f_3 = rectangle('Position',[imdb.images.target_bboxs(now_index,[1,2])+1,...
                        imdb.images.target_bboxs(now_index,[3,4])-imdb.images.target_bboxs(now_index,[1,2])]);
                    subplot(1,2,2),f_2 = imshow(imread(imdb.images.search{now_index}));hold on ;
                    f_4 = rectangle('Position',[imdb.images.search_bboxs(now_index,[1,2])+1,...
                        imdb.images.search_bboxs(now_index,[3,4])-imdb.images.search_bboxs(now_index,[1,2])]);
                    title('search');
                    drawnow;
                else
                    f_1.set('CData',imread(imdb.images.target{now_index}));
                    f_2.set('CData',imread(imdb.images.search{now_index}));
                    f_3.set('Position',[imdb.images.target_bboxs(now_index,[1,2])+1,...
                        imdb.images.target_bboxs(now_index,[3,4])-imdb.images.target_bboxs(now_index,[1,2])]);
                    f_4.set('Position',[imdb.images.search_bboxs(now_index,[1,2])+1,...
                        imdb.images.search_bboxs(now_index,[3,4])-imdb.images.search_bboxs(now_index,[1,2])]);
                    drawnow;
                end
            end
        end
        now_index = now_index+1;
    end %%end frame
end %%end v


if opts.lite
    imdb.images.target = imdb.images.target(lite_index);
    imdb.images.search = imdb.images.search(lite_index);
    imdb.images.set = imdb.images.set(lite_index);
    imdb.images.target_bboxs = imdb.images.target_bboxs(lite_index,:);
    imdb.images.search_bboxs = imdb.images.search_bboxs(lite_index,:);
end
dataMean(1,1,1:3) = single([123,117,104]);
imdb.images.data_mean(1,1,1:3) = dataMean;
imdb.meta.sets = {'train', 'val'} ;
end %%end function



