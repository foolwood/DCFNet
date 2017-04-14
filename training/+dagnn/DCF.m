classdef DCF < dagnn.ElementWise
%DCF  layer
%   Discriminant Correlation Filters(DCF)
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
            
            xf = fft2(inputs{1});% target region
            zf = fft2(inputs{2});% search region
            xf_conj = conj(xf);
            [h,w,c,~] = size(xf);
            hwc = h*w*c;
            
            useGPU = isa(xf, 'gpuArray');
            if isempty(obj.yf)
                obj.initYF(useGPU);
            end
            
            kxxf = sum(xf .* xf_conj, 3) ./ hwc;
            alphaf = bsxfun(@rdivide, obj.yf, (kxxf + obj.lambda));
            kzxf = sum(zf .* xf_conj, 3) ./ hwc;
            outputs{1} = real(ifft2(alphaf .* kzxf));
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            
            dldrf = fft2(derOutputs{1}); 
            xf = fft2(inputs{1});% target region
            zf = fft2(inputs{2});% search region
            xf_conj = conj(xf);
            
            [h,w,c,~] = size(xf);
            hwc = h*w*c;
            
            kxxf = sum(xf .* xf_conj, 3) ./ hwc +obj.lambda;
            
            alphaf = bsxfun(@rdivide,obj.yf, kxxf);
            dldz = real(ifft2(bsxfun(@times,dldrf.*conj(alphaf),xf)))/hwc;
            kzxf = sum(zf .* xf_conj, 3) ./ hwc;
            dldx = real(ifft2(bsxfun(@times,conj(dldrf).*alphaf,zf)-...
            2*bsxfun(@times,xf,real(dldrf.*conj(alphaf.*kzxf)./kxxf))))/hwc;
            
            derInputs{1} = dldx;
            derInputs{2} = dldz;
            derParams = {};
        end
        
        function initYF(obj, useGPU)
            yf_ = single(fft2(gaussian_shaped_labels(obj.sigma, obj.win_size)));
            lambda_ = gather(obj.lambda);
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