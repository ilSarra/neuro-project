clearvars -except sub_id subjects erd_ers_gdf erd_ers_psd psd_computing model_training model_testing subjects_selected class_model;

%% PSD computing

% Data information

datapath = 'micontinuous/';
chan_label = {'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C3', 'C1', 'Cz', 'C2', 'C4', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4'};
class_ID = [771 773 783];   
class_label = {'both hands', 'both feet', 'rest'};
nclasses = length(class_ID);
mod_ID = [0 1];
mod_label = {'offline', 'online'};

% PSD parameters

mlength = 1;
wlength = 0.5;
pshift = 0.25;                  
wshift = 0.0625;  
selfreqs = 4:2:96;
winconv = 'backward';

% Fetch subjects' directories from datapath

files_in_datapath = dir(datapath);
folders_in_datapath = files_in_datapath([files_in_datapath.isdir]);
subjects = {folders_in_datapath(3:end).name};   % start from 3 to skip . and ..


disp(['PSD computing for subject ' sub_id{1}]);

files = dir(fullfile([datapath '/' sub_id{1}], '*.gdf'));
nfiles = size(files,1);

load('laplacian16.mat');
all_psd = [];
runs = [];
TYP = [];
DUR = [];
POS = [];

for i = 1:nfiles
    filename = strcat(datapath, sub_id{1}, '/', files(i).name);
    disp(['Loading file ' filename]);

    [current_s, current_h] = sload(filename);

    current_s = current_s(:, 1:size(chan_label,2));
    sample_rate = current_h.SampleRate;

    % Compute laplacian filter

    current_s = current_s * lap;

    % Compute PSD

    [psd, freq_grid] = proc_spectrogram(current_s, wlength, wshift, pshift, sample_rate, mlength);

    % Select frequencies

    [freqs, idfreqs] = intersect(freq_grid, selfreqs);
    psd = psd(:, idfreqs, :);

    % Extract events

    current_events = current_h.EVENT;
    win_events.TYP = current_events.TYP;
    win_events.POS = proc_pos2win(current_events.POS, wshift * sample_rate, winconv, mlength * sample_rate);
    win_events.DUR = floor(current_events.DUR/(wshift * sample_rate)) + 1;
    win_events.conversion = winconv;

    info.mlength  = mlength;
    info.wlength  = wlength;
    info.pshift   = pshift;
    info.wshift   = wshift;
    info.selfreqs = selfreqs;
    info.winconv  = winconv;

    % Export .mat file

    psd_file = strcat(filename(1:size(filename,2) - 3),'mat');
    disp(['Saving psd in: ' psd_file]);
    save(psd_file, 'psd', 'freqs', 'win_events', 'info', 'sample_rate');
end