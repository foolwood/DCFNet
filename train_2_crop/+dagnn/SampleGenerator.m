%DAGNN.SampleGenerator Generate sample for GOTURN
% input:
%     -bbox_prev_gt       1x4
%     -bbox_curr_gt       1x4
%     -image_prev      HxWx3x1
%     -image_curr      HxWx3x1
% output:
%     -bbox_gt_scaled           1x 1x4xNo
%     -image_target_pad       HoxWox3xNo
%     -image_search_pad       HoxWox3xNo
%   2016 Qiang Wang
classdef SampleGenerator < dagnn.Layer
    
    properties
        Ho = 0;
        Wo = 0;
        No = 1;
        padding = 1;
        visual = true;
    end
    
    properties (Transient)
        % the grid --> this is cached
        % has the size: [2 x HoWo]
        xxyy ;
        averageImage = reshape(single([123,117,104]),[1,1,3]);
    end
    
    methods
        
        function outputs = forward(obj, inputs, ~)
            bbox_prev_gt = inputs{1};
            bbox_curr_gt = inputs{2};
            image_prev = inputs{3};
            image_curr = inputs{4};
            
            % generate the grid coordinates:
            useGPU = isa(bbox_prev_gt, 'gpuArray');
            if isempty(obj.xxyy)
                obj.initGrid(useGPU);
            end
            
            [im_h,im_w,im_c,~] = size(image_prev);
            if im_c == 1
                image_prev = repmat(image_prev,[1,1,3,1]);
            end
            
            %% target
            target_crop_w = (1+obj.padding)*(bbox_prev_gt(3)-bbox_prev_gt(1));
            target_crop_h = (1+obj.padding)*(bbox_prev_gt(4)-bbox_prev_gt(2));
            target_crop_cx = (bbox_prev_gt(3)+bbox_prev_gt(1))/2;
            target_crop_cy = (bbox_prev_gt(4)+bbox_prev_gt(2))/2;
            search_crop_cx = (bbox_curr_gt(3)+bbox_curr_gt(1))/2;
            search_crop_cy = (bbox_curr_gt(4)+bbox_curr_gt(2))/2;
            
            cy_t = (target_crop_cy*2/(im_h-1))-1;
            cx_t = (target_crop_cx*2/(im_w-1))-1;
            
            h_s = target_crop_h/(im_h-1);
            w_s = target_crop_w/(im_w-1);
            
            s = reshape([h_s;w_s], 2,1,1); % x,y scaling
            t = reshape([cy_t;cx_t], 2,1,1); % translation
            
            g = bsxfun(@times, obj.xxyy, s); % scale
            g = bsxfun(@plus, g, t); % translate
            g = reshape(g, 2,obj.Ho,obj.Wo,1);
            
            target_pad = vl_nnbilinearsampler(image_prev, single(g));
            
            image_target_pad = repmat(target_pad,[1,1,1,obj.No]);
            image_search_pad = vl_nnbilinearsampler(image_curr, single(g));
            
            if useGPU,
                delta_yx_scaled = gpuArray(zeros([obj.No,2],'single'));%buff
            else
                delta_yx_scaled = zeros([obj.No,2],'single');%buff
            end
            
            curr_target_location = [target_crop_cy,target_crop_cx];
            curr_search_location = [search_crop_cy,search_crop_cx];
            delta_yx_scaled(1,1:2) = ...
                (curr_search_location-curr_target_location).*...
                [obj.Ho,obj.Wo]./[target_crop_h,target_crop_w];
            
            %% search
            if obj.No > 1
                delta_xy_rand_shift = rand(obj.No-1,2,'single')*0.6-0.3;
                delta_xy_rand_shift = repmat(...
                    bsxfun(@times,bbox_prev_gt([3,4])-bbox_prev_gt([1,2]),delta_xy_rand_shift),[1,2]);
                bbox_curr_shift = bsxfun(@plus,bbox_prev_gt,delta_xy_rand_shift);
                
                target_crop_w = (1+obj.padding)*(bbox_curr_shift(:,3)-bbox_curr_shift(:,1))';
                target_crop_h = (1+obj.padding)*(bbox_curr_shift(:,4)-bbox_curr_shift(:,2))';
                target_crop_cx = (bbox_curr_shift(:,1)+bbox_curr_shift(:,3))'/2;
                target_crop_cy = (bbox_curr_shift(:,2)+bbox_curr_shift(:,4))'/2;
                
                delta_yx_scaled(2:obj.No,2:-1:1) = ...
                    (-delta_xy_rand_shift(:,1:2)).*...
                    (bsxfun(@rdivide,[obj.Wo,obj.Ho],[target_crop_w',target_crop_h']));
                
                cy_t = (target_crop_cy*2/(im_h-1))-1;
                cx_t = (target_crop_cx*2/(im_w-1))-1;
                
                h_s = target_crop_h/(im_h-1);
                w_s = target_crop_w/(im_w-1);
                
                s = reshape([h_s;w_s], 2,1,[]); % x,y scaling
                t = reshape([cy_t;cx_t], 2,1,[]); % translation
                
                g = bsxfun(@times, obj.xxyy, s); % scale
                g = bsxfun(@plus, g, t); % translate
                g = reshape(g, 2,obj.Ho,obj.Wo,[]);
                
                image_search_pad(:,:,:,2:obj.No) = vl_nnbilinearsampler(image_curr, single(g));
            end
            
            if obj.visual
                close all;
                for i = 1:obj.No
                    subplot(4,obj.No/4,i);imshow(uint8(image_search_pad(:,:,:,i)));hold on;
                    plot(delta_yx_scaled(i,2)+obj.Wo/2,delta_yx_scaled(i,1)+obj.Ho/2,'r*');
                end
                drawnow;
            end
            
            image_target_pad = bsxfun(@minus,image_target_pad,obj.averageImage);
            image_search_pad = bsxfun(@minus,image_search_pad,obj.averageImage);
            outputs = {image_target_pad,image_search_pad,round(delta_yx_scaled)};
        end
        
        function obj = SampleGenerator(varargin)
            obj.load(varargin);
            % get the output sizes:
            obj.Ho = obj.Ho;
            obj.Wo = obj.Wo;
            obj.No = obj.No;
            obj.padding = obj.padding;
            obj.xxyy = [];
            obj.averageImage = obj.averageImage;
            obj.visual = obj.visual;
        end
        
        function obj = reset(obj)
            reset@dagnn.Layer(obj) ;
            obj.xxyy = [] ;
        end
        
        function initGrid(obj, useGPU)
            % initialize the grid:
            % this is a constant
            xi = linspace(-1, 1, obj.Ho);
            yi = linspace(-1, 1, obj.Wo);
            [yy,xx] = meshgrid(yi,xi);
            xxyy_ = [xx(:), yy(:)]' ; % 2xM
            if useGPU
                obj.xxyy = gpuArray(xxyy_);
                obj.averageImage = gpuArray(obj.averageImage);
            end
            obj.xxyy = xxyy_ ;
        end
    end
end
