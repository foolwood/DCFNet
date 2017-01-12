
% error('Tracker not configured! Please edit the tracker_DCFNet.m file.'); % Remove this line after proper configuration

% The human readable label for the tracker, used to identify the tracker in reports
% If not set, it will be set to the same value as the identifier.
% It does not have to be unique, but it is best that it is.
tracker_label = ['DCFNetn'];

% For MATLAB implementations we have created a handy function that generates the appropritate
% command that will run the matlab executable and execute the given script that includes your
% tracker implementation.
%
% Please customize the line below by substituting the first argument with the name of the
% script (not the .m file but just the name of the script as you would use it in within Matlab)
% of your tracker and also provide the path (or multiple paths) where the tracker sources
% are found as the elements of the cell array (second argument).
net_index = 6;
interp_factor = 0.01;
numScale = 3;
output_sigma_factor = 0.1;
scale_penalty = 0.97;
param = ['''net_index'',' num2str(net_index) ','...
    '''interp_factor'',' num2str(interp_factor) ','...
    '''numScale'',' num2str(numScale) ','...
    '''scale_penalty'',' num2str(scale_penalty) ','...
     '''visual'',' 'false' ','...
     '''gpu'',' 'false' ','...
    '''output_sigma_factor'',' num2str(output_sigma_factor)];

tracker_command = generate_matlab_command(['DCFNet_vot(',param,')'], {'C:\Users\qiangwang\Documents\GitHub\DCFNet\track4vot'});

tracker_interpreter = 'matlab';

% tracker_linkpath = {}; % A cell array of custom library directories used by the tracker executable (optional)

% tracker_trax = false; % Uncomment to manually disable TraX protocol testing
