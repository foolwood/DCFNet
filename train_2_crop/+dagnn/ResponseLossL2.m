classdef ResponseLossL2 < dagnn.Loss
%ResponseLossL2  layer
%  `ResponseLossL2.forward({r, x*})` computes L2 loss.
%
%  Here the Loss between two matrices is defined as:
%
%     Loss = |r - r_idea|.*|r - r_idea|
%
%  Input 
%       - r    w x h x 1 x N.
%       - x0   N x 2 
%   QiangWang, 2016
% -------------------------------------------------------------------------------------------------------------------------
    properties
        win_size = [3,3];
        sigma = 1;
    end
    properties (Transient)
        ny = [];
    end
    methods
        function outputs = forward(obj, inputs, params)
            assert(numel(inputs) == 2, 'two inputs are needed');
            r = inputs{1}; % r
            
            delta_yx = inputs{2}; % delta_xy
            delta_x = mod(delta_yx(:,2),obj.win_size(2))+1;% 1-index
            delta_y = mod(delta_yx(:,1),obj.win_size(1))+1;% 1-index
            
            delta_yx_ind = sub2ind(obj.win_size,delta_y,delta_x);
            
            useGPU = isa(r, 'gpuArray');
            if isempty(obj.ny)
                obj.initNY(useGPU);
            end
           
            loss = (r - obj.ny(:,:,1,delta_yx_ind));
            loss = loss.*loss;
            
%             subplot(2,2,1);imagesc(r(:,:,1)); subplot(2,2,2);imagesc(obj.ny(:,:,1,delta_yx_ind(1)));
%             subplot(2,2,4);imagesc(loss(:,:,1));colorbar();
%             drawnow
            
            outputs{1} = sum(sum(sum(sum(loss))))/size(r,4);
            
            n = obj.numAveraged ;
            m = n + 1 ;
            obj.average = (n * obj.average + gather(outputs{1})) / m ;
            obj.numAveraged = m ;

        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            assert(numel(inputs) == 2, 'two inputs are needed');
            r = inputs{1}; % r
            
            delta_yx = inputs{2}; % delta_xy
            delta_x = mod(delta_yx(:,2),obj.win_size(2))+1;% 1-index
            delta_y = mod(delta_yx(:,1),obj.win_size(1))+1;% 1-index
            
            delta_yx_ind = sub2ind(obj.win_size,delta_y,delta_x);
            r_idea = obj.ny(:,:,1,delta_yx_ind);
            
            derInputs = {(derOutputs{1}*2)*(r - r_idea), []} ;
            derParams = {} ;
        end
        
        function initNY(obj, useGPU)
            ny_ = zeros([obj.win_size,1,obj.win_size],'single');
            
            y_ = single(gaussian_shaped_labels(obj.sigma, obj.win_size));
            
            for i = 1:obj.win_size(1)
                for j = 1:obj.win_size(2)
                    ny_(:,:,1,i,j) = circshift(y_,[i-1,j-1]);
                end
            end
            if useGPU
                obj.ny = gpuArray(ny_);
            else
                obj.ny = ny_;
            end
        end
        
        function obj = reset(obj)
            obj.ny = [] ;
        end
        
        function obj = ResponseLossL2(varargin)
            obj.load(varargin);
            obj.win_size = obj.win_size;
            obj.sigma = obj.sigma ;
            obj.ny = [];
        end

    end

end


function labels = gaussian_shaped_labels(sigma, sz)%kcf
[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
labels = exp(-0.5 / sigma^2 * (rs.^2 + cs.^2));
labels = circshift(labels, -floor(sz(1:2) / 2) + 1);
assert(labels(1,1) == 1)
end
