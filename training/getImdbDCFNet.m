function imdb = getImdbDCFNet(varargin)
opts = [];
opts.dataDir = fullfile('..','data');

opts.visualization = false;
opts.output_size = 125;
opts.padding = 1.5;
opts = vl_argparse(opts, varargin);

bbox_mode = 'axis_aligned';
if strcmp(bbox_mode,'axis_aligned')
    get_bbox = @get_axis_aligned_BB;
elseif strcmp(bbox_mode,'minmax')
    get_bbox = @get_minmax_BB;
end
% -------------------------------------------------------------------------
%   full dataset:
%           vid2015: 464873
% -------------------------------------------------------------------------
set_name = {'vid2015'};
num_all_frame = 464873;
%% Be careful!!! It takes a HUGE RAM for fast speed!
imdb.images.set = int8(ones(1, num_all_frame));
imdb.images.set(randperm(num_all_frame, 1000)) = int8(2);
imdb.images.images = zeros(opts.output_size, opts.output_size, 3, num_all_frame, 'uint8');
imdb.images.up_index = zeros(1, num_all_frame, 'double'); % The farthest frame can it touch
now_index = 0;

% -------------------------------------------------------------------------
%                                                                   VID2015
% -------------------------------------------------------------------------
if any(strcmpi(set_name, 'vid2015'))
    disp('VID2015 Data:');
    if exist('vid_2015_seg.mat', 'file')
        load('vid_2015_seg.mat');%% use dataPreprocessing;
    else
        error('You should generate <vid_2015_seg.mat> according ''dataPreprocessing'' at first.')
    end
    videos = seg;
    n_videos = numel(videos);
    for  v = 1:n_videos
        video = videos{v};fprintf('%3d / %3d \n', v, n_videos);
        
        img_files = video.path;
        
        ground_truth = cell2mat((video.rect)');
        ground_truth_4xy = [ground_truth(:,1),ground_truth(:,2),...
            ground_truth(:,1),ground_truth(:,2)+ground_truth(:,4),...
            ground_truth(:,1)+ground_truth(:,3),ground_truth(:,2)+ground_truth(:,4),...
            ground_truth(:,1)+ground_truth(:,3),ground_truth(:,2)];
        invaildindex = any(isnan(ground_truth), 2);
        img_files(invaildindex) = [];
        ground_truth_4xy(invaildindex,:) = [];
        
        im_frist = vl_imreadjpeg(img_files(1));
        [H, W, ~] = size(im_frist{1});
        im_bank = vl_imreadjpeg(img_files, 'Pack', 'Resize', [H, W], 'numThreads', 32);
        bbox_gt = get_bbox(ground_truth_4xy);
        n_frames = size(bbox_gt,1);
        imdb.images.images(:,:,:,now_index+(1:n_frames)) = uint8(...
            imcrop_pad(im_bank{1}, bbox_gt, opts.padding, opts.output_size([1,1])));
        imdb.images.up_index(now_index+(1:n_frames)) = (n_frames:-1:1)-1;
        imdb.images.set(now_index+n_frames) = 4; %should not be selected as x.
        now_index = now_index + n_frames;
    end %%end v
end %%end VID2017

% dataMean = single(mean(imdb.images.images,4));
dataMean(1,1,1:3) = single([123,117,104]);
imdb.images.data_mean(1, 1, 1:3) = dataMean;
imdb.meta.sets = {'train', 'val'} ;
end %%end function
