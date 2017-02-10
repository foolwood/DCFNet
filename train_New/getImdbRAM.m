function imdb = getImdbRAM(varargin)
addpath('../utils');
opts = [];
opts.dataDir = '../data';

opts.vot16_dataDir = fullfile(opts.dataDir,'VOT16');
opts.otb_dataDir = fullfile(opts.dataDir,'OTB');
opts.nus_pro_dataDir = fullfile(opts.dataDir,'NUS_PRO');
opts.tc128_dataDir = fullfile(opts.dataDir,'Temple-color-128');
opts.alov300_dataDir = fullfile(opts.dataDir,'ALOV300');
opts.uav123_dataDir = fullfile(opts.dataDir,'UAV123');
addpath(opts.uav123_dataDir);

opts.visualization = false;
opts.output_size = [125,125];
opts.padding = 1.5;
opts.dataset = 2;
opts = vl_argparse(opts, varargin);

bbox_mode = 'axis_aligned';
if strcmp(bbox_mode,'axis_aligned')
    get_bbox = @get_axis_aligned_BB;
elseif strcmp(bbox_mode,'minmax')
    get_bbox = @get_minmax_BB;
end
% -------------------------------------------------------------------------
%   full dataset:
%           vot16:21395+60 vot15:21395 vot14:10188 vot13:5665
%           cvpr2013:29435 tb100:58935 tb50:26922
%           nus_pro:26090+73 tc128:55217+129 alov300:89351 det16:478806
%           uav123:
%   Special dataset:
%           alov300_goturn:15570
% -------------------------------------------------------------------------

switch opts.dataset
    case 1
        set_name = {'vot16'};
        num_all_frame = 21395+60;
    case 2
        set_name = {'nus_pro','tc128'};
        num_all_frame = (26090+73)+(55217+129);
    case 3
        set_name = {'nus_pro','tc128_ce','uav123'};
        num_all_frame = (26090+73)+(30507+78)+(109772+123);
    otherwise
        error('No such version!');
end

imdb.images.set = int8(ones(1,num_all_frame));
imdb.images.set(randperm(num_all_frame,100)) = int8(2);
imdb.images.images = zeros(125,125,3,num_all_frame,'uint8');
imdb.images.up_index = zeros(1,num_all_frame,'double');
now_index = 0;
% -------------------------------------------------------------------------
%                                                                     VOT16
% -------------------------------------------------------------------------
if any(strcmpi(set_name,'vot16'))
    disp('VOT2016 Data:');
    vot16_dataDir = opts.vot16_dataDir;
    videos = importdata(fullfile(vot16_dataDir,'list.txt'));
    n_videos = numel(videos);
    for v = 1:n_videos
        video = videos{v};fprintf('%3d / %3d :%20s\n',v,n_videos,video);
        [img_files, ground_truth_4xy] = load_video_info_vot(vot16_dataDir, video);
        im_frist = vl_imreadjpeg(img_files(1));
        [H,W,~] = size(im_frist{1});
        im_bank = vl_imreadjpeg(img_files,'Pack','Resize',[H,W]);
        bbox_gt = get_bbox(ground_truth_4xy);
        n_frames = size(bbox_gt,1);
        imdb.images.images(:,:,:,now_index+(1:n_frames)) = uint8(...
            imcrop_my(im_bank{1},bbox_gt,opts.padding,opts.output_size));
        imdb.images.up_index(now_index+(1:n_frames)) = (n_frames-1):-1:0;
        imdb.images.set(now_index+n_frames) = 4;%should not be selected.
        now_index = now_index+n_frames;
    end %%end v
end

% -------------------------------------------------------------------------
%                                                                   NUS_PRO
% -------------------------------------------------------------------------
if any(strcmpi(set_name,'nus_pro'))
    
    disp('NUS_PRO Data:');
    nus_pro_dataDir = opts.nus_pro_dataDir;
    filename = fullfile(nus_pro_dataDir,'seq_list_with_gt.csv');
    videos = importdata(filename);
    n_videos = numel(videos);
    for  v = 1:n_videos
        video = videos{v};fprintf('%3d / %3d :%20s\n',v,n_videos,video);
        [img_files, ground_truth_4xy] = load_video_info_nus_pro(nus_pro_dataDir, video);
        im_frist = vl_imreadjpeg(img_files(1));
        [H,W,~] = size(im_frist{1});
        im_bank = vl_imreadjpeg(img_files,'Pack','Resize',[H,W]);
        bbox_gt = get_bbox(ground_truth_4xy);
        n_frames = size(bbox_gt,1);
        imdb.images.images(:,:,:,now_index+(1:n_frames)) = uint8(...
            imcrop_my(im_bank{1},bbox_gt,opts.padding,opts.output_size));
        imdb.images.up_index(now_index+(1:n_frames)) = (n_frames-1):-1:0;
        imdb.images.set(now_index+n_frames) = 4;%should not be selected.
        now_index = now_index+n_frames;
    end %%end v
end %%end nus-pro
% -------------------------------------------------------------------------
%                                                                     TC128
% -------------------------------------------------------------------------
if any(strcmpi(set_name,'tc128'))
    
    disp('TC128 Data:');
    tc128_dataDir = opts.tc128_dataDir;
    TC128_temp = dir(tc128_dataDir);
    TC128 = {TC128_temp.name};
    TC128(strcmp('.', TC128) | strcmp('..', TC128)| ~[TC128_temp.isdir]) = [];
    videos = TC128;
    n_videos = numel(videos);
    for  v = 1:n_videos
        video = videos{v};fprintf('%3d / %3d :%20s\n',v,n_videos,video);
        [img_files, ground_truth_4xy] = load_video_info_tc128(tc128_dataDir, video);
        im_frist = vl_imreadjpeg(img_files(1));
        [H,W,~] = size(im_frist{1});
        im_bank = vl_imreadjpeg(img_files,'Pack','Resize',[H,W]);
        bbox_gt = get_bbox(ground_truth_4xy);
        n_frames = size(bbox_gt,1);
        imdb.images.images(:,:,:,now_index+(1:n_frames)) = uint8(...
            imcrop_my(im_bank{1},bbox_gt,opts.padding,opts.output_size));
        imdb.images.up_index(now_index+(1:n_frames)) = (n_frames-1):-1:0;
        imdb.images.set(now_index+n_frames) = 4;%should not be selected.
        now_index = now_index+n_frames;
    end %%end v
end %%end tc128
% -------------------------------------------------------------------------
%                                                                     TC128
% -------------------------------------------------------------------------
if any(strcmpi(set_name,'tc128_ce'))
    
    disp('TC128ce Data:');
    tc128_dataDir = opts.tc128_dataDir;
    TC128_temp = dir(tc128_dataDir);
    TC128 = {TC128_temp.name};
    TC128(strcmp('.', TC128) | strcmp('..', TC128)| ~[TC128_temp.isdir]) = [];
    tc128_no_cvpr2013_index = zeros(1,numel(TC128));
    for k = 1:numel(TC128)
        if numel(strfind(TC128{k},'_ce'))~=0
            tc128_no_cvpr2013_index(k) = 1;
        else
            tc128_no_cvpr2013_index(k) = 0;
        end
    end
    videos = TC128(tc128_no_cvpr2013_index==1);
    n_videos = numel(videos);
    for  v = 1:n_videos
        video = videos{v};fprintf('%3d / %3d :%20s\n',v,n_videos,video);
        [img_files, ground_truth_4xy] = load_video_info_tc128(tc128_dataDir, video);
        im_frist = vl_imreadjpeg(img_files(1));
        [H,W,~] = size(im_frist{1});
        im_bank = vl_imreadjpeg(img_files,'Pack','Resize',[H,W]);
        bbox_gt = get_bbox(ground_truth_4xy);
        n_frames = size(bbox_gt,1);
        imdb.images.images(:,:,:,now_index+(1:n_frames)) = uint8(...
            imcrop_my(im_bank{1},bbox_gt,opts.padding,opts.output_size));
        imdb.images.up_index(now_index+(1:n_frames)) = (n_frames-1):-1:0;
        imdb.images.set(now_index+n_frames) = 4;%should not be selected.
        now_index = now_index+n_frames;
    end %%end v
end %%end tc128_ce

% -------------------------------------------------------------------------
%                                                                    UAV123
% -------------------------------------------------------------------------
if any(strcmpi(set_name,'uav123'))
    
    disp('UAV123 Data:');
    uav123_dataDir = opts.uav123_dataDir;
    uav123_annoDir = fullfile(uav123_dataDir,'anno','UAV123');
    videos = configSeqs;
    n_videos = numel(videos);
    for  v = 1:n_videos
        video = videos{v};fprintf('%3d / %3d :%20s\n',v,n_videos,video.name);
        
        nz	= strcat('%0',num2str(video.nz),'d'); %number of zeros in the name of image
        img_files = num2str((video.startFrame : video.endFrame)', [nz,'.',video.ext]);
        img_files = cellstr(img_files);
        img_files = fullfile(video.path,img_files);
        
        ground_truth = dlmread(fullfile(uav123_annoDir,[video.name '.txt']));
        ground_truth_4xy = [ground_truth(:,1),ground_truth(:,2),...
            ground_truth(:,1),ground_truth(:,2)+ground_truth(:,4),...
            ground_truth(:,1)+ground_truth(:,3),ground_truth(:,2)+ground_truth(:,4),...
            ground_truth(:,1)+ground_truth(:,3),ground_truth(:,2)];
        invaildindex = any(isnan(ground_truth),2);
        img_files(invaildindex) = [];
        ground_truth_4xy(invaildindex,:) = [];
        
        im_frist = vl_imreadjpeg(img_files(1));
        [H,W,~] = size(im_frist{1});
        im_bank = vl_imreadjpeg(img_files,'Pack','Resize',[H,W]);
        bbox_gt = get_bbox(ground_truth_4xy);
        n_frames = size(bbox_gt,1);
        imdb.images.images(:,:,:,now_index+(1:n_frames)) = uint8(...
            imcrop_my(im_bank{1},bbox_gt,opts.padding,opts.output_size));
        imdb.images.up_index(now_index+(1:n_frames)) = (n_frames-1):-1:0;
        imdb.images.set(now_index+n_frames) = 4;%should not be selected.
        now_index = now_index+n_frames;
    end %%end v
end %%end uav123

% dataMean = single(mean(imdb.images.images,4));
dataMean(1,1,1:3) = single([123,117,104]);
imdb.images.data_mean(1,1,1:3) = dataMean;
imdb.meta.sets = {'train', 'val'} ;
end %%end function
