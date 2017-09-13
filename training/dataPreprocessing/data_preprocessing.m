function data_preprocessing()
%Data Preprocessiong for structured storage of VID.
%
% Output:
% 
%      subset
%          - video
%                - frame
%                      - image_height
%                      - image_width
%                      - path
%                      - object
%                           - c
%                           - occluded
%                           - rect
%                           - trackid
% 
%
%   version 1.0 --July/2017
%
%   Written by Qiang Wang (qiangwang2015 AT ia.ac.cn)
%
clear; close all; clc; clear all;
visualization = 0;

project_path = fileparts(fileparts(fileparts(mfilename('fullpath'))));
anno_path = fullfile(project_path, 'data/ILSVRC/Annotations/VID/train/');
data_path = fullfile(project_path, 'data/ILSVRC/Data/VID/train/');

meta_file = 'meta_vid.mat';
load(meta_file);
hash = make_hash(synsets);
colormap = hsv(30);

set_name = dir(fullfile(anno_path, 'ILSVRC2015_VID_train_*'));
set_name = sort({set_name.name});

for set_id = 1:numel(set_name)
    subset = [];
    anno_subset_path = fullfile(anno_path, set_name{set_id});
    video_name = dir(fullfile(anno_subset_path, 'ILSVRC2015_train_*'));
    video_name = sort({video_name.name});
    
    for video_id = 1:numel(video_name)
        video = [];
        disp([set_name{set_id} '/' num2str(video_id, '%08d')]);
        pathSrcXml = fullfile(anno_subset_path, video_name{video_id});
        xml_name = dir(fullfile(pathSrcXml, '*.xml'));
        xml_name = sort({xml_name.name});
        
        for xml_id = 1:numel(xml_name)
            frame = [];
            res = VOCreadxml(fullfile(pathSrcXml, xml_name{xml_id}));
            
            frame.image_width = str2double(res.annotation.size.width);
            frame.image_height = str2double(res.annotation.size.height);
            frame.path = fullfile(data_path, res.annotation.folder, [res.annotation.filename '.JPEG']);
            
            if visualization
                imagesc(imread(frame.path));
                set(gca,'XTick',[],'YTick',[]);
                hold on;
            end
            
            object = [];
            if isfield(res.annotation, 'object')
                object = cell(1,numel(res.annotation.object));
                for k=1:numel(res.annotation.object)
                    obj = res.annotation.object(k);
                    object{k}.c = get_class2node(hash, obj.name);
                    b = obj.bndbox;
                    bb = str2double({b.xmin b.ymin b.xmax b.ymax});
                    object{k}.rect = [bb(1), bb(2), bb(3) - bb(1), bb(4) - bb(2)];
                    object{k}.occluded = str2double(obj.occluded);
                    object{k}.trackid = str2double(obj.trackid) + 1; % 1-index
                    
                    if visualization
                        if object{k}.occluded
                            rectangle('position', object{k}.rect, 'LineWidth', 3, ...
                                'EdgeColor', colormap(object{k}.c,:),'LineStyle',':');
                        else
                            rectangle('position', object{k}.rect, 'LineWidth', 3, ...
                                'EdgeColor', colormap(object{k}.c,:));
                        end
                        drawnow; hold off;
                    end
                end
            end
            frame.object = object;
            video.frame{xml_id} = frame;
        end
        subset.video{video_id} = video;
    end
    save(['vid_subset', num2str(set_id), '.mat'], 'subset');
end
end