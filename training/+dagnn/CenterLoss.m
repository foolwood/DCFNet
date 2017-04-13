classdef CenterLoss < dagnn.Loss
    
    properties
        win_size = [125,125];
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            r = inputs{1};
            
            center_loss = 0;
            for i = 1:size(r,4)
                response = r(:,:,i);
                [vert_delta, horiz_delta] = find(response == max(response(:)), 1);
                if vert_delta > obj.win_size(1) / 2  %wrap around to negative half-space of vertical axis
                    vert_delta = vert_delta - obj.win_size(1);
                end
                if horiz_delta > obj.win_size(2) / 2  %same for horizontal axis
                    horiz_delta = horiz_delta - obj.win_size(2);
                end
                delta_yx_pred = [vert_delta-1, horiz_delta-1];
                center_loss = center_loss+norm(delta_yx_pred);
            end
            outputs{1} = center_loss;
            
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + gather(outputs{1})) / m ;
            obj.numAveraged = m ;
        end
        
        function reset(obj)
            obj.average = 0 ;
            obj.numAveraged = 0 ;
        end
        
        function obj = CenterLoss(varargin)
            obj.load(varargin) ;
            obj.win_size = obj.win_size;
        end
    end
end