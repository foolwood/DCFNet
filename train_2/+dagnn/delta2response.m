classdef delta2response < dagnn.Filter
%DCF  layer
%   Dual Correlation Filter(DCF) two activations of same size exploiting  the API of vl_nnconv
%
%   QiangWang, 2016
% -------------------------------------------------------------------------------------------------------------------------
    properties
        win_size = [3,3];
        sigma = 1;
        y = [];
        ny = [];
    end

    methods
        function outputs = forward(obj, inputs, params)
            assert(numel(inputs) == 1, 'one inputs are needed');

            delta_xy = inputs{1}; % delta_xy
            n = size(delta_xy,1);
            if isempty(obj.ny)
                obj.ny = repmat(obj.y,[1,1,1,n]);
            end
            for i = 1:n
                obj.ny(:,:,1,i) = circshift(obj.y,delta_xy(i,:));
            end
            outputs{1} = obj.ny;

        end

        function obj = delta2response(varargin)
            obj.load(varargin);
            obj.win_size = obj.win_size;
            obj.sigma = obj.sigma ;
            obj.y = gaussian_shaped_labels(obj.sigma, obj.win_size);
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

