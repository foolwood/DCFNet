classdef ResponseLossSmoothL1 < dagnn.Loss
%ResponseLossSmoothL1  layer
%  `ResponseLossSmoothL1.forward({r, x*})` computes relu loss,
%  weighting the elements by the smooth L1 distance 
%  between `x` and `x*`.
%
%  Here the Loss between two vectors is defined as:
%
%     Loss = sum_i (relu(r_i - r_idea))^2* f(x_i-x*).
%
%  where f is the function :
%
%              { 0.1,       if |delta| < 15,
%   f(delta) = {
%              { 1,         otherwise.
%
%  Input 
%       - r    w x h x 1 x N.
%       - x0   N x 2 
%   QiangWang, 2016
% -------------------------------------------------------------------------------------------------------------------------
    properties
        win_size = [3,3];
        sigma = 1;
        threshold = 0.1;
        ny = [];
        nf = [];
    end

    methods
        function outputs = forward(obj, inputs, params)
            assert(numel(inputs) == 2, 'two inputs are needed');
            r = squeeze(inputs{1}); % r
            
            delta_yx = inputs{2}; % delta_xy
            delta_x = mod(delta_yx(:,2),obj.win_size(2))+1;% 1-index
            delta_y = mod(delta_yx(:,1),obj.win_size(1))+1;% 1-index
            
            delta_yx_ind = sub2ind(obj.win_size,delta_y,delta_x);
            r_idea = obj.ny(:,:,delta_yx_ind);
            
            f = obj.nf(:,:,delta_yx_ind);
            loss = max(r - r_idea,0).*max(r - r_idea,0).*f;
            
%             subplot(2,2,1);imagesc(r); subplot(2,2,2);imagesc(r_idea);
%             subplot(2,2,3);imagesc(f); subplot(2,2,4);imagesc(loss);
%             drawnow

%             outputs{1} = reshape(loss,obj.win_size(1),obj.win_size(2),1,[]);
            
            outputs{1} = sum(loss(:));
            
            n = obj.numAveraged ;
            m = n + 1 + 1e-9 ;
            obj.average = (n * obj.average + gather(outputs{1})) / m ;
            obj.numAveraged = m ;

        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            assert(numel(inputs) == 2, 'two inputs are needed');
            r = squeeze(inputs{1}); % r
            
            delta_yx = inputs{2}; % delta_xy
            delta_x = mod(delta_yx(:,2),obj.win_size(1))+1;% 1-index
            delta_y = mod(delta_yx(:,1),obj.win_size(2))+1;% 1-index
            
            delta_xy_ind = sub2ind(obj.win_size,delta_y,delta_x);
            r_idea = obj.ny(:,:,delta_xy_ind);
            
            delta = (max(r - r_idea,0)).*obj.nf(:,:,delta_xy_ind) ;
            
            derInputs = {reshape(delta,obj.win_size(1),obj.win_size(2),1,[])...
                .* derOutputs{1}, []} ;
            derParams = {} ;
        end

        function obj = ResponseLossSmoothL1(varargin)
            obj.load(varargin);
            obj.win_size = obj.win_size;
            obj.sigma = obj.sigma ;
            obj.threshold = obj.threshold;
            obj.ny = zeros([obj.win_size,obj.win_size],'single');
            obj.nf = zeros([obj.win_size,obj.win_size],'single');
            
            y_ = single(gaussian_shaped_labels(obj.sigma, obj.win_size));
            f_ = single(1 - 0.9* (y_ > obj.threshold));
            
            for i = 1:obj.win_size(1)
                for j = 1:obj.win_size(2)
                    obj.ny(:,:,i,j) = circshift(y_,[i-1,j-1]);
                    obj.nf(:,:,i,j) = circshift(f_,[i-1,j-1]);
                end
            end
        end

    end

end


function labels = gaussian_shaped_labels(sigma, sz)%kcf
[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
labels = exp(-0.5 / sigma^2 * (rs.^2 + cs.^2));
labels = circshift(labels, -floor(sz(1:2) / 2) + 1);
assert(labels(1,1) == 1)
end




% function test()
% x = rand(16,16,16,16);
% z = x;
% yf = repmat(fft2(gaussian_shaped_labels(1, [16,16])),[1,1,1,16]);
% lambda = 1e-4;
% 
% zf = fft2(z);
% xf = fft2(x);
% kxxf = sum(xf .* conj(xf), 3) / numel(xf(:,:,:,1));
% alphaf = yf ./ (kxxf + lambda);
% kzf = sum(xf .* conj(zf), 3) / numel(xf(:,:,:,1));
% responses = real(ifft2(alphaf .* kzf));
% end
