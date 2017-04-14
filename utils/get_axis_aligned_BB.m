% -------------------------------------------------------------------------------------------------
function [bb] = get_axis_aligned_BB(region)
%GETAXISALIGNEDBB computes axis-aligned bbox with same area as the rotated one (REGION)
% -------------------------------------------------------------------------------------------------
cx = mean(region(:,1:2:end),2);
cy = mean(region(:,2:2:end),2);
x1 = min(region(:,1:2:end),[],2);
x2 = max(region(:,1:2:end),[],2);
y1 = min(region(:,2:2:end),[],2);
y2 = max(region(:,2:2:end),[],2);
x1y1x2y2 = region(:,1:2) - region(:,3:4);
x2y2x3y3 = region(:,3:4) - region(:,5:6);
A1 = sqrt(sum(x1y1x2y2.*x1y1x2y2,2)).* sqrt(sum(x2y2x3y3.*x2y2x3y3,2));
A2 = (x2 - x1) .* (y2 - y1);
s = sqrt(A1./A2);
w = s .* (x2 - x1);
h = s .* (y2 - y1);
bb = [cx-w/2,cy-h/2,cx+w/2,cy+h/2]-1;
end
