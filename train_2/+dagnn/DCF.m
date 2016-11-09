classdef DCF < dagnn.Layer
%DCF  layer
%   Dual Correlation Filter(DCF) two activations of same size exploiting  the API of vl_nnconv
%
%   QiangWang, 2016
% -------------------------------------------------------------------------------------------------------------------------
    properties
        opts = {'cuDNN'}
        yf = fft2(gaussian_shaped_labels(1, [227,227]));
    end

    methods
        function outputs = forward(obj, inputs, params)
            assert(numel(inputs) == 2, 'two inputs are needed');

            z = inputs{1}; % target
            x = inputs{2}; % search region

            assert(ndims(z) == ndims(x), 'z and x have different number of dimensions');
            assert(size(z,1) == size(x,1), 'z and x have different size');
            
            zf = fft2(z);
            xf = fft2(x);
            kf = sum(zf .* conj(zf), 3) / numel(zf);
            alphaf = obj.yf ./ (kf + lambda); 
            kzf = sum(xf .* conj(zf), 3) / numel(xf);
            outputs{1} = real(ifft2(alphaf .* kzf));

        end

        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            assert(numel(inputs) == 2, 'two inputs are needed');
            assert(numel(derOutputs) == 1, 'only one gradient should be flowing in this layer (dldy)');
            z = inputs{1}; % exemplar
            x = inputs{2}; % instance
            assert(size(z,1) < size(x,1), 'exemplar z has to be smaller than instance x');
            [wx,hx,cx,bx] = size(x);
            x = reshape(x, [wx,hx,cx*bx,1]);
            dldy = derOutputs{1};
            [wdl,hdl,cdl,bdl] = size(dldy);
            assert(cdl==1);
            dldy = reshape(dldy, [wdl,hdl,cdl*bdl,1]);
            [dldx, dldz, ~] = vl_nnconv(x, z, [], dldy);
            [mx,nx,cb,one] = size(dldx);
            assert(mx == size(x, 1));
            assert(nx == size(x, 2));
            assert(cb == cx * bx);
            assert(one == 1);
            derInputs{1} = dldz;
            derInputs{2} = reshape(dldx, [mx,nx,cx,bx]);
            derParams = {};
        end

        function outputSizes = getOutputSizes(obj, inputSizes)
            z_sz = inputSizes{1};
            x_sz = inputSizes{2};
            y_sz = [x_sz(1:2) - z_sz(1:2) + 1, 1, z_sz(4)];
            outputSizes = {y_sz};
        end

        function rfs = getReceptiveFields(obj)
            rfs(1,1).size = [inf inf]; % could be anything
            rfs(1,1).stride = [1 1];
            rfs(1,1).offset = 1;
            rfs(2,1).size = [inf inf];
            rfs(2,1).stride = [1 1];
            rfs(2,1).offset = 1;
        end

        function obj = DCF(varargin)
            obj.load(varargin);
        end

    end

end



function labels = gaussian_shaped_labels(sigma, sz)
%GAUSSIAN_SHAPED_LABELS
%   Gaussian-shaped labels for all shifts of a sample.
%
%   LABELS = GAUSSIAN_SHAPED_LABELS(SIGMA, SZ)
%   Creates an array of labels (regression targets) for all shifts of a
%   sample of dimensions SZ. The output will have size SZ, representing
%   one label for each possible shift. The labels will be Gaussian-shaped,
%   with the peak at 0-shift (top-left element of the array), decaying
%   as the distance increases, and wrapping around at the borders.
%   The Gaussian function has spatial bandwidth SIGMA.
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/


% 	%as a simple example, the limit sigma = 0 would be a Dirac delta,
% 	%instead of a Gaussian:
% 	labels = zeros(sz(1:2));  %labels for all shifted samples
% 	labels(1,1) = magnitude;  %label for 0-shift (original sample)


%evaluate a Gaussian with the peak at the center element
[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
labels = exp(-0.5 / sigma^2 * (rs.^2 + cs.^2));

%move the peak to the top-left, with wrap-around
labels = circshift(labels, -floor(sz(1:2) / 2) + 1);

%sanity check: make sure it's really at top-left
assert(labels(1,1) == 1)

end

