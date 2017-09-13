function analyze_data()
clear; close all; clc; clear all;
load('vid_subset1.mat');allset{1} = subset;
load('vid_subset2.mat');allset{2} = subset;
load('vid_subset3.mat');allset{3} = subset;
load('vid_subset4.mat');allset{4} = subset;

% forbidden_classes = {'lizard','snake','train','whale'};
% ok = ~ismember(class, forbidden_classes);
% {' please do not confuse with snake (lizards have legs)','lizard','n01674464',17}
% {' please do not confuse with lizard (snakes do not have legs)','snake','n01726692',23}
% {[],'train','n04468005',26}
% {[],'whale','n02062744',29}

object_count = 0;
seg_count = 0;

for set_id = 1:numel(allset)
    sub_set = allset{set_id};
    for video_id = 1:numel(sub_set.video)
        video = sub_set.video{video_id};
        object_table = [];
        for frame_id = 1:numel(video.frame)
            frame = video.frame{frame_id};
            for object_id = 1:numel(frame.object)
                object = frame.object{object_id};
                rect = object.rect;

                if (object.c ~= 17 && object.c ~= 23 && object.c ~= 26 && object.c ~= 29 &&...
                        object.occluded == 0 &&...
                        checkSize([frame.image_width, frame.image_height], rect) &&...
                        checkBorders([frame.image_width, frame.image_height], rect))
                    object_table(frame_id, frame.object{object_id}.trackid) = 1;
                end
            end
        end
        
        updown = [0,1,0;0,1,0;0,1,0];%Remove isolated point
        if(size(object_table, 1) > 1)
            object_table = conv2(object_table, updown, 'same') > 1 & object_table;
            cc = bwconncomp(object_table, updown);
            n_seg = cc.NumObjects;
            sum_object = sum(object_table(:));
            object_count = object_count + sum_object;
            
            for o = 1:n_seg
                [frame_id, track_id] = ind2sub(cc.ImageSize, cc.PixelIdxList{1, o});
                frame_id = frame_id';
                track_id = unique(track_id);
                fff = 0;
                for f = frame_id
                    seg{seg_count+o}.path{fff+1} = video.frame{f}.path;
                    
                    for object = video.frame{f}.object
                        if(object{1}.trackid == track_id)
                            seg{seg_count+o}.rect{fff+1} = object{1}.rect;
                        end
                    end
                    fff = fff+1;
                end
            end
            seg_count = seg_count + n_seg;
            fprintf('set:%d\t video:%d\t object:%d\t seg:%d\t sum:%d\t sum_seg:%d\n',...
                set_id, video_id, sum_object, n_seg, object_count, seg_count);
        end
    end
end
save(fullfile('..', 'vid_2015_seg'), 'seg');

%% VID Statistical info
video_count = 0;
frame_count = 0;
object_count = 0;
boundingbox_count = 0;
for set_id = 1:numel(allset)
    sub_set = allset{set_id};
    video_count = video_count + numel(sub_set.video);
    for video_id = 1:numel(sub_set.video)
        video = sub_set.video{video_id};
        object_table = [];
        frame_count = frame_count + numel(video.frame);
        for frame_id = 1:numel(video.frame)
            frame = video.frame{frame_id};
            for object_id = 1:numel(frame.object)
                object_table(end+1) = frame.object{object_id}.trackid;
            end
        end
        object_count = object_count + numel(unique(object_table));
        boundingbox_count = boundingbox_count + numel(object_table);
    end
end
fprintf('\n\n\n\n VID 2015 training set info: \n\n')
fprintf('\t\t %d snippets \n\t\t %d frames \n\t\t %d objects \n\t\t %d boundingboxes\n',...
    video_count, frame_count, object_count, boundingbox_count);
end


%% From SiameseFC
function ok = checkSize(frame_sz, object_extent)
min_ratio = 0.1;
max_ratio = 0.75;
% accept only objects >10% and <75% of the total frame
area_ratio = sqrt((object_extent(3)*object_extent(4))/prod(frame_sz));
ok = area_ratio > min_ratio && area_ratio < max_ratio;
end

function ok = checkBorders(frame_sz, object_extent)
dist_from_border = 0.05 * (object_extent(3) + object_extent(4))/2;
ok = object_extent(1) > dist_from_border && object_extent(2) > dist_from_border && ...
    (frame_sz(1)-(object_extent(1)+object_extent(3))) > dist_from_border && ...
    (frame_sz(2)-(object_extent(2)+object_extent(4))) > dist_from_border;
end

