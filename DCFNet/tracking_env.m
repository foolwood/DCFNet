function tracking_env()
addpath('../matconvnet/matlab');
run('vl_setupnn.m') ;
fftw('planner','patient');
end