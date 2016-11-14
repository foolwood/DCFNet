classdef LossL2 < dagnn.Loss
%LossL2  L2 loss
%  `LossL2.forward({x, x0})` computes the  L2 distance
%  between `x` and `x0`.
%
%  Here the smooth L2 loss between two vectors is defined as:
%
%     Loss = sum_i f(x_i - x0_i).
%
%  where f is the function:
%
%   f(delta) = |delta|.*|delta|
%
%  In practice, `x` and `x0` as h x w x c x n arrays.

methods
    function outputs = forward(obj, inputs, params)

        subplot(1,2,1);imagesc(inputs{1}(:,:,1,1));
        subplot(1,2,2);imagesc(inputs{2}(:,:,1,1));
        drawnow;
        delta = inputs{1} - inputs{2} ;
        l2Delta = (delta.*delta) ;

        outputs{1} = sum(l2Delta(:))/numel(delta) ;

        % Accumulate loss statistics.
        n = obj.numAveraged ;
        m = n + 1 + 1e-9 ;
        obj.average = (n * obj.average + gather(outputs{1})) / m ;
        obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        % Function derivative:
        %
        %  f'(x) = 2x
        %

        delta = 2*(inputs{1} - inputs{2}) ;
        derInputs = {delta .* derOutputs{1}, []} ;
        derParams = {} ;
    end

    function obj = LossL1(varargin)
        obj.load(varargin) ;
        obj.loss = 'l2';
    end
end
end