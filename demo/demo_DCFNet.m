function res = demo_DCFNet()
addpath(fullfile('..','DCFNet'));

init_rect = [129,80,64,78];
img_file = dir('./David/img/*.jpg');
img_file = fullfile('./David/img/', {img_file.name});
subS.init_rect = init_rect;
subS.s_frames = img_file;

param = [];
param.gpu = true;
gpuDevice(1);
param.visual = true;

res = run_DCFNet(subS,0,0,param);
disp(['fps: ',res.fps]);

end