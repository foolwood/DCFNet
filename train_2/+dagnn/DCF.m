classdef DCF < dagnn.ElementWise
%DCF  layer
%   Dual Correlation Filter(DCF) two activations of same size
%
%   QiangWang, 2016
% -------------------------------------------------------------------------------------------------------------------------
    properties
        win_size = [3,3];
        sigma = 1;
    end
    properties (Transient)
        yf = [];
        lambda = 1e-4;
    end
    methods
        function outputs = forward(obj, inputs, params)
            assert(numel(inputs) == 2, 'two inputs are needed');
            
            x = inputs{1}; % target region
            z = inputs{2}; % search region
            xf = fft2(x);
            zf = fft2(z);
            xf_conj = conj(xf);
            [h,w,c,~] = size(x);
            mn = h*w*c;
            
            assert(ndims(z) == ndims(x), 'z and x have same number of dimensions');
            assert(all(size(z) == size(x)), 'z and x have same size');
            
            useGPU = isa(x, 'gpuArray');
            if isempty(obj.yf)
                obj.initYF(useGPU);
            end
            
            kxxf = sum(xf .* xf_conj, 3) ./ mn;
            alphaf = bsxfun(@rdivide,obj.yf,(kxxf + obj.lambda)); 
            kzxf = sum(zf .* xf_conj, 3) ./ mn;
            outputs{1} = real(ifft2(alphaf .* kzxf));
        end

        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            assert(numel(inputs) == 2, 'two inputs are needed');
            assert(numel(derOutputs) == 1, 'only one gradient should be flowing in this layer (dldy)');
           
            dldr = derOutputs{1};
            
            x = inputs{1}; % target region
            z = inputs{2}; % search region
            xf = fft2(x);
            zf = fft2(z);
            xf_conj = conj(xf);
            
            [h,w,c,~] = size(x);
            mn = h*w*c;
            
            kxxf = sum(xf .* xf_conj, 3) / mn;
            alphaf = bsxfun(@rdivide,obj.yf,(kxxf + obj.lambda)); 
            
            dldrf = fft2(dldr);
            dldz = real(ifft2(-bsxfun(@times,dldrf.*alphaf,xf_conj)/mn));
%             dldz =[];
            dldx = real(ifft2(conj(bsxfun(@times,dldrf,...
                bsxfun(@rdivide,...
                bsxfun(@times,conj(bsxfun(@times,zf,obj.yf)/mn),kxxf+obj.lambda)-...
                bsxfun(@times,bsxfun(@times,(sum(zf.*xf_conj,3)/mn),obj.yf),xf_conj/mn),...
                (kxxf + obj.lambda).*(kxxf + obj.lambda))))));
%             dldx = [];
            derInputs{1} = dldx;
            derInputs{2} = dldz;
            derParams = {};
        end
        
        function initYF(obj, useGPU)
            yf_ = single(fft2(gaussian_shaped_labels(obj.sigma, obj.win_size)));
            lambda_ = 1e-4;
            if useGPU
                obj.yf = gpuArray(yf_);
                obj.lambda = gpuArray(lambda_);
            else
                obj.yf = yf_;
                obj.lambda = lambda_;
            end
        end
        
        function obj = reset(obj)
            obj.yf = [] ;
            obj.lambda = 1e-4;
        end
        
        function obj = DCF(varargin)
            obj.load(varargin);
            obj.win_size = obj.win_size;
            obj.sigma = obj.sigma ;
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
