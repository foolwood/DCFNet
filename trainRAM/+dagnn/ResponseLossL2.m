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
        y = [];
    end
    methods
        function outputs = forward(obj, inputs, params)
            r = inputs{1}; % r
            
            useGPU = isa(r, 'gpuArray');
            if isempty(obj.y)
                obj.initY(useGPU);
            end
           
            loss = bsxfun(@minus,r,obj.y);
            loss = loss.*loss;
            
            outputs{1} = sum(loss(:))/size(r,4);
            
            n = obj.numAveraged ;
            m = n + 1 ;
            obj.average = (n * obj.average + gather(outputs{1})) / m ;
            obj.numAveraged = m ;

        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            r = inputs{1}; % r
            derInputs{1} = (derOutputs{1}*2/size(r,4)).*bsxfun(@minus,r,obj.y) ;
            derParams = {} ;
        end
        
        function initY(obj, useGPU)
            
            y_ = single(gaussian_shaped_labels(obj.sigma, obj.win_size));

            if useGPU
                obj.y = gpuArray(y_);
            else
                obj.y = y_;
            end
        end
        
        function obj = reset(obj)
            obj.y = [] ;
            obj.average = 0 ;
            obj.numAveraged = 0 ;
        end
        
        function obj = ResponseLossL2(varargin)
            obj.load(varargin);
            obj.win_size = obj.win_size;
            obj.sigma = obj.sigma ;
            obj.y = [];
        end

    end

end


function labels = gaussian_shaped_labels(sigma, sz)%kcf
[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
labels = exp(-0.5 / sigma^2 * (rs.^2 + cs.^2));
labels = circshift(labels, -floor(sz(1:2) / 2) + 1);
assert(labels(1,1) == 1)
end
