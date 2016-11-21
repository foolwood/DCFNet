load wind
xmin = min(x(:));
xmax = max(x(:));
ymax = max(y(:));
zmin = min(z(:));

wind_speed = sqrt(u.^2 + v.^2 + w.^2);
% hsurfaces = slice(x,y,z,wind_speed,[xmin,100,xmax],ymax,zmin);
% hsurfaces = slice(x,y,z,wind_speed,[xmin,100,xmax],[],[]);
hsurfaces = slice(x,y,z,wind_speed,[xmin,100,xmax],ymax,zmin);
set(hsurfaces,'FaceColor','interp','EdgeColor','none')
colormap jet

hcont = ...
contourslice(x,y,z,wind_speed,[xmin,100,xmax],[],[]);
set(hcont,'EdgeColor',[0.7 0.7 0.7],'LineWidth',0.5)
xp = zeros(1,size(wind_speed,3));
yp = zeros(1,size(wind_speed,3));
zp = zeros(1,size(wind_speed,3));

for i = 1:size(wind_speed,2)
    wind_speed_temp = squeeze(wind_speed(:,i,:));
    [delta_y,delta_z] = find(wind_speed_temp == max(max(wind_speed_temp)));
    xp(i) = x(delta_y,i,delta_z);
    yp(i) = y(delta_y,i,delta_z);
    zp(i) = z(delta_y,i,delta_z);
end
hold on;
xp = spline(1:numel(xp),xp,1:0.1:numel(xp));
yp = spline(1:numel(yp),yp,1:0.1:numel(yp));
zp = spline(1:numel(zp),zp,1:0.1:numel(zp));
hlines = plot3(xp,yp,zp);
set(hlines,'LineWidth',10,'Color','r');

plot3(xp([1,end]),yp([1,end]),zp([1,end]),...
    'MarkerFaceColor',[1 0 0],'MarkerEdgeColor',[1 0 0],...
    'MarkerSize',50,...
    'Marker','.',...
    'LineStyle','none',...
    'Color',[0 0 1]);


% plot3(squeeze(x(:,1,:)),squeeze(y(:,1,:)),squeeze(z(:,1,:)),'r*');

view(3)
daspect([2,2,1])
axis tight
axis off

set(gca, 'Position', get(gca, 'OuterPosition') - ...
    get(gca, 'TightInset') * [-1 0 1 0; 0 -1 0 1; 0 0 1 0; 0 0 0 1]);

saveas(gcf,'visual_training','pdf');
saveas(gcf,'visual_training','eps');