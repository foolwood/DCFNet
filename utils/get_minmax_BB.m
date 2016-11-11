% -------------------------------------------------------------------------------------------------
function [bb] = get_minmax_BB(region)
%get_minmax_BB computes minmax bbox from the rotated one (REGION)
% -------------------------------------------------------------------------------------------------

bb = [min(region(:,1:2:end),[],2),...
    min(region(:,2:2:end),[],2),...
    max(region(:,1:2:end),[],2),...
    max(region(:,2:2:end),[],2)]-1;

end
