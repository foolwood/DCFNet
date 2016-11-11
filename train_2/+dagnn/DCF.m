classdef DCF < dagnn.Filter
%DCF  layer
%   Dual Correlation Filter(DCF) two activations of same size exploiting  the API of vl_nnconv
%
%   QiangWang, 2016
% -------------------------------------------------------------------------------------------------------------------------
    properties
        win_size = [3,3];
        sigma = 1;
        yf = [];
        lambda = 1e-4;
        alphaf = [];
        xf = [];
    end

    methods
        function outputs = forward(obj, inputs, params)
            assert(numel(inputs) == 2, 'two inputs are needed');

            x = inputs{1}; % target region
            z = inputs{2}; % search region

            assert(ndims(z) == ndims(x), 'z and x have different number of dimensions');
            assert(size(z,1) == size(x,1), 'z and x have different size');
            
            zf = fft2(z);
            obj.xf = fft2(x);
            kxxf = sum(obj.xf .* conj(obj.xf), 3) / numel(obj.xf(:,:,:,1));
            obj.alphaf = bsxfun(@rdivide,obj.yf,(kxxf + obj.lambda)); 
            kzf = sum( zf.* conj(obj.xf), 3) / numel(obj.xf(:,:,:,1));
            outputs{1} = real(ifft2(obj.alphaf .* kzf));

        end

        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            assert(numel(inputs) == 2, 'two inputs are needed');
            assert(numel(derOutputs) == 1, 'only one gradient should be flowing in this layer (dldy)');
           
            dldr = derOutputs{1};
            
            dldrf = fft2(dldr);
            dldkf = dldrf.*obj.alphaf;
            dldzf = bsxfun(@times,dldkf,conj(obj.xf))/ numel(obj.xf(:,:,:,1));
            dldz = real(ifft2(dldzf));
            dldx = [];
            derInputs{1} = dldx;
            derInputs{2} = dldz;
            derParams = {};
        end
        
        function obj = DCF(varargin)
            obj.load(varargin);
            obj.win_size = obj.win_size;
            obj.sigma = obj.sigma ;
            obj.yf = fft2(gaussian_shaped_labels(obj.sigma, obj.win_size));
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
