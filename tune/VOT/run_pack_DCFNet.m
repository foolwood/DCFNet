% This script can be used to pack the results and submit them to a challenge.
function run_pack_DCFNet(name)
addpath('F:\vot-toolkit'); toolkit_path; % Make sure that VOT toolkit is in the path

[sequences, experiments] = workspace_load();

tracker = tracker_load(name);

workspace_submit(tracker, sequences, experiments);
end

