classdef CCenterLoss < dagnn.Loss
    
    properties
        win_size = [125,125];
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            r = inputs{1};
            delta_yx = inputs{2};
            
            center_loss = 0;
            for i = 1:size(delta_yx,1)
                response = r(:,:,i);
                [vert_delta, horiz_delta] = find(response == max(response(:)), 1);
                
                vert_delta = vert_delta - floor(obj.win_size(1)/2);
                horiz_delta = horiz_delta - floor(obj.win_size(2)/2);
                
                delta_yx_pred = [vert_delta-1, horiz_delta-1];
                center_loss = center_loss+norm(delta_yx_pred - delta_yx(i,:));
            end
            outputs{1} = center_loss;
            
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + gather(outputs{1})) / m ;
            obj.numAveraged = m ;
        end
        
        function obj = CCenterLoss(varargin)
            obj.load(varargin) ;
            obj.win_size = obj.win_size;
        end
    end
end