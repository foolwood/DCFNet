v = VideoWriter('feature_response.avi','Uncompressed AVI');
open(v);
for f = 1:1000
    feature = rand(125,125,16,'single');
    response = rand(480,480,'single');
    a = subplot(4,8,[1,2,3,4,8,9,10,11,12,16,17,18,19,24,25,26,27]);imagesc(response);axis off;axis equal tight;
    set(a,'position',[0,0,0.5,1]);
    for i = [1,4,2,3]
        for j = 1:4
            a = subplot(4,8,(i-1)*8+9-j);
            imagesc(feature(:,:,(i-1)*4+j));axis off;axis equal tight;
            set(a,'position',[1-j*0.125+0.025/2,1-i*0.25+0.05/2,0.1,0.2]);
        end
    end
    pause(0.01)
    writeVideo(v,getframe(gcf));
    
    
end
close(v);
