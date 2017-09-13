function training_env()
if isunix()
    addpath('/home/qwang/matconvnet/matlab')    
else
    addpath('D:\matconvnet-1.0-beta24\matlab')
end

run('vl_setupnn.m') ;

addpath(fullfile('..','utils'));

fftw('planner','patient');
end