
%
%  High-Speed Tracking with Kernelized Correlation Filters
%
%  Joao F. Henriques, 2014
%  http://www.isr.uc.pt/~henriques/
%
%  Main interface for Kernelized/Dual Correlation Filters (KCF/DCF).
%  This function takes care of setting up parameters, loading video
%  information and computing precisions. For the actual tracking code,
%  check out the TRACKER function.
%
%  RUN_TRACKER
%    Without any parameters, will ask you to choose a video, track using
%    the Gaussian KCF on HOG, and show the results in an interactive
%    figure. Press 'Esc' to stop the tracker early. You can navigate the
%    video using the scrollbar at the bottom.
%
%  RUN_TRACKER VIDEO
%    Allows you to select a VIDEO by its name. 'all' will run all videos
%    and show average statistics. 'choose' will select one interactively.
%
%  RUN_TRACKER VIDEO KERNEL
%    Choose a KERNEL. 'gaussian'/'polynomial' to run KCF, 'linear' for DCF.
%
%  RUN_TRACKER VIDEO KERNEL FEATURE
%    Choose a FEATURE type, either 'hog' or 'gray' (raw pixels).
%
%  RUN_TRACKER(VIDEO, KERNEL, FEATURE, SHOW_VISUALIZATION, SHOW_PLOTS)
%    Decide whether to show the scrollable figure, and the precision plot.
%
%  Useful combinations:
%  >> run_tracker choose gaussian hog  %Kernelized Correlation Filter (KCF)
%  >> run_tracker choose linear hog    %Dual Correlation Filter (DCF)
%  >> run_tracker choose gaussian gray %Single-channel KCF (ECCV'12 paper)
%  >> run_tracker choose linear gray   %MOSSE filter (single channel)
%


function [precision, fps] = run_tracker(video, kernel_type, feature_type, show_visualization, show_plots)

%path to the videos (you'll be able to choose one with the GUI).
base_path = '../data/OTB/';

%default settings
if nargin < 1, video = 'choose'; end
if nargin < 2, kernel_type = 'linear'; end
if nargin < 3, feature_type = 'rand'; end
if nargin < 4, show_visualization = ~strcmp(video, 'all'); end
if nargin < 5, show_plots = ~strcmp(video, 'all'); end


%parameters according to the paper. at this point we can override
%parameters based on the chosen kernel or feature type
kernel.type = kernel_type;

features.gray = false;
features.hog = false;
features.rand = false;

padding = 1.5;  %extra area surrounding the target
lambda = 1e-4;  %regularization
output_sigma_factor = 0.1;  %spatial bandwidth (proportional to target)

switch feature_type
    case 'gray',
        interp_factor = 0.075;  %linear interpolation factor for adaptation
        
        kernel.sigma = 0.2;  %gaussian kernel bandwidth
        
        kernel.poly_a = 1;  %polynomial kernel additive term
        kernel.poly_b = 7;  %polynomial kernel exponent
        
        features.gray = true;
        cell_size = 1;
        
    case 'hog',
        interp_factor = 0.02;
        
        kernel.sigma = 0.5;
        
        kernel.poly_a = 1;
        kernel.poly_b = 9;
        
        features.hog = true;
        features.hog_orientations = 9;
        cell_size = 4;
    case 'rand',
        interp_factor = 0.02;
        
        kernel.sigma = 0.5;
        
        kernel.poly_a = 1;
        kernel.poly_b = 9;
        
        features.rand = true;
        cell_size = 1;
        
    otherwise
        error('Unknown feature.')
end


assert(any(strcmp(kernel_type, {'linear', 'polynomial', 'gaussian'})), 'Unknown kernel.')


switch video
    case 'choose',
        %ask the user for the video, then call self with that video name.
        video = choose_video(base_path);
        if ~isempty(video),
            [precision, fps] = run_tracker(video, kernel_type, ...
                feature_type, show_visualization, show_plots);
            
            if nargout == 0,  %don't output precision as an argument
                clear precision
            end
        end
        
        
    case 'all',
        %all videos, call self with each video name.
        
        %only keep valid directory names
        dirs = dir(base_path);
        videos = {dirs.name};
        videos(strcmp('.', videos) | strcmp('..', videos) | ...
            strcmp('anno', videos) | ~[dirs.isdir]) = [];
        
        %the 'Jogging' sequence has 2 targets, create one entry for each.
        %we could make this more general if multiple targets per video
        %becomes a common occurence.
        videos(strcmpi('Jogging', videos)) = [];
        videos(end+1:end+2) = {'Jogging.1', 'Jogging.2'};
        videos(strcmpi('Human4', videos)) = [];
        videos(end+1:end+1) = {'Human4.2'};
        videos(strcmpi('Skating2', videos)) = [];
        videos(end+1:end+2) = {'Skating2.1', 'Skating2.2'};
        
        all_precisions = zeros(numel(videos),1);  %to compute averages
        all_fps = zeros(numel(videos),1);
        
        if ~exist('matlabpool', 'file'),
            %no parallel toolbox, use a simple 'for' to iterate
            for k = 1:numel(videos),
                [all_precisions(k), all_fps(k)] = run_tracker(videos{k}, ...
                    kernel_type, feature_type, show_visualization, show_plots);
            end
        else
            %evaluate trackers for all videos in parallel
            if matlabpool('size') == 0,
                matlabpool open;
            end
            parfor k = 1:numel(videos),
                [all_precisions(k), all_fps(k)] = run_tracker(videos{k}, ...
                    kernel_type, feature_type, show_visualization, show_plots);
            end
        end
        
        %compute average precision at 20px, and FPS
        mean_precision = mean(all_precisions);
        fps = mean(all_fps);
        fprintf('\nAverage precision (20px):% 1.3f, Average FPS:% 4.2f\n\n', mean_precision, fps)
        if nargout > 0,
            precision = mean_precision;
        end
        
        
    case 'benchmark',
        %running in benchmark mode - this is meant to interface easily
        %with the benchmark's code.
        
        %get information (image file names, initial position, etc) from
        %the benchmark's workspace variables
        seq = evalin('base', 'subS');
        target_sz = seq.init_rect(1,[4,3]);
        pos = seq.init_rect(1,[2,1]) + floor(target_sz/2);
        img_files = seq.s_frames;
        video_path = [];
        
        %call tracker function with all the relevant parameters
        positions = tracker(video_path, img_files, pos, target_sz, ...
            padding, kernel, lambda, output_sigma_factor, interp_factor, ...
            cell_size, features, false);
        
        %return results to benchmark, in a workspace variable
        rects = [positions(:,2) - target_sz(2)/2, positions(:,1) - target_sz(1)/2];
        rects(:,3) = target_sz(2);
        rects(:,4) = target_sz(1);
        res.type = 'rect';
        res.res = rects;
        assignin('base', 'res', res);
        
        
    otherwise
        %we were given the name of a single video to process.
        
        %get image file names, initial state, and ground truth for evaluation
        [img_files, pos, target_sz, ground_truth, video_path] = load_video_info(base_path, video);
        
        
        %call tracker function with all the relevant parameters
        [positions, time] = tracker(video_path, img_files, pos, target_sz, ...
            padding, kernel, lambda, output_sigma_factor, interp_factor, ...
            cell_size, features, show_visualization);
        
        
        %calculate and show precision plot, as well as frames-per-second
        precisions = precision_plot(positions, ground_truth, video, show_plots);
        fps = numel(img_files) / time;
        
        fprintf('%12s - Precision (20px):% 1.3f, FPS:% 4.2f\n', video, precisions(20), fps)
        
        if nargout > 0,
            %return precisions at a 20 pixels threshold
            precision = precisions(20);
        end
        
end
end
