clearvars -except sub_id subjects erd_ers_gdf erd_ers_psd psd_computing model_training model_testing subjects_selected class_model;

%% Importing GDF data (offline runs)

% Data information

datapath = 'micontinuous/';
chan_label = {'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C3', 'C1', 'Cz', 'C2', 'C4', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4'};
class_ID = [771 773 783];   
class_label = {'both hands', 'both feet', 'rest'};
mod_ID = [0 1];
mod_label = {'offline', 'online'};


% Fetch subjects' directories from datapath

files_in_datapath = dir(datapath);
folders_in_datapath = files_in_datapath([files_in_datapath.isdir]);
subjects = {folders_in_datapath(3:end).name};   % start from 3 to skip . and ..

disp(['ERD/ERS on GDF for subject ' sub_id{1}]);

files = dir(fullfile([datapath '/' sub_id{1}], '*.gdf'));
nfiles = size(files,1);

all_s = [];
runs = [];
TYP = [];
DUR = [];
POS = [];

for i = 1:nfiles

    filename = strcat(datapath, sub_id{1}, '/', files(i).name);

    if(contains(filename,'offline'))
        
        disp(['Loading file ' filename]);

        [current_s, current_h] = sload(filename);
        
        current_s = current_s(:, 1:size(chan_label,2));
        

        sample_rate = current_h.SampleRate;
        
        events = current_h.EVENT;

        TYP = cat(1, TYP, events.TYP);
        DUR = cat(1, DUR, events.DUR);
        POS = cat(1, POS, events.POS + size(all_s, 1));

        all_s = cat(1, all_s, current_s);

        current_run = i * ones(size(current_s, 1), 1);
        runs = cat(1, runs, current_run);
    end

end

% Extracting information from data

nsamples  = size(all_s, 1);
nchannels = size(all_s, 2);

CFeedbackPOS = POS(TYP == 781);
CFeedbackDUR = DUR(TYP == 781);

CuePOS = POS(TYP == 771 | TYP == 773 | TYP == 783);
CueDUR = DUR(TYP == 771 | TYP == 773 | TYP == 783);
CueTYP = TYP(TYP == 771 | TYP == 773 | TYP == 783);

FixPOS = POS(TYP == 786);
FixDUR = DUR(TYP == 786);
FixTYP = TYP(TYP == 786);

ntrials = length(CFeedbackPOS);

cue = zeros(nsamples, 1);
trials = zeros(nsamples, 1);
TrialStart = nan(ntrials, 1);
TrialStop  = nan(ntrials, 1);
FixStart = nan(ntrials, 1);
FixStop  = nan(ntrials, 1);

for i = 1:ntrials
    current_start = CuePOS(i);
    current_stop  = CFeedbackPOS(i) + CFeedbackDUR(i) - 1;
    cue(current_start:current_stop) = CueTYP(i);
    trials(current_start:current_stop) = i;
    
    TrialStart(i) = current_start;
    TrialStop(i)  = current_stop;
    FixStart(i) = FixPOS(i);
    FixStop(i) = FixPOS(i) + FixDUR(i) - 1;
end

%% Data processing

% Spatial filters

load('laplacian16.mat');

s_lap = all_s * lap;
all_s = s_lap;

% Creating filters

filtOrder = 4;
band_mu   = [8 12];
band_beta = [18 22]; 

% Filter parameters

[b_mu, a_mu] = butter(filtOrder, band_mu*2/sample_rate);
[b_beta, a_beta] = butter(filtOrder, band_beta*2/sample_rate);

% Applying filters

all_s_mu = zeros(size(all_s));
all_s_beta = zeros(size(all_s));

for i = 1:nchannels
    all_s_mu(:, i) = filtfilt(b_mu, a_mu, all_s(:, i));
    all_s_beta(:, i) = filtfilt(b_beta, a_beta, all_s(:, i));
end

% Squaring

srect_mu = power(all_s_mu, 2);
srect_beta = power(all_s_beta, 2);

% Moving average

smovavg_mu = zeros(size(all_s));
smovavg_beta = zeros(size(all_s));

for i = 1:nchannels
    smovavg_mu(:, i) = (filter(ones(1, sample_rate)/ ...
                        sample_rate, 1, srect_mu(:, i)));
    smovavg_beta(:, i) = (filter(ones(1, sample_rate)/ ...
                        sample_rate, 1, srect_beta(:, i)));
end


% Logarithmic transformation

slogpower_mu = (smovavg_mu);
slogpower_beta = (smovavg_beta);

% Trial extraction

min_trial_dur = min(TrialStop - TrialStart);
trials_data_mu   = nan(min_trial_dur, nchannels, ntrials);
trials_data_beta = nan(min_trial_dur, nchannels, ntrials);
trial_cue = zeros(ntrials, 1);
trial_run = zeros(ntrials, 1);

for i = 1:ntrials
    current_start = TrialStart(i);
    current_stop  = current_start + min_trial_dur - 1;

    trials_data_mu(:, :, i)   = slogpower_mu(current_start:current_stop, :);
    trials_data_beta(:, :, i) = slogpower_beta(current_start:current_stop, :);

    trial_cue(i) = unique(cue(current_start:current_stop));
    trial_run(i) = unique(runs(current_start:current_stop));
end

% Baseline extraction (from fixation)

min_fix_dur = min(FixStop - FixStart);
fixation_mu   = nan(min_fix_dur, nchannels, ntrials);
fixation_beta = nan(min_fix_dur, nchannels, ntrials);

for i = 1:ntrials

    fixation_mu(:, :, i) = slogpower_mu(FixStart(i):FixStart(i) ...
                             + min_fix_dur - 1, :);
    fixation_beta(:, :, i) = slogpower_beta(FixStart(i):FixStart(i) ...
                             + min_fix_dur - 1, :);
end

%% ERD/ERS

disp('Computing ERD/ERS on GDF data');

% Average and replicate the value of the baseline

baseline_mu   = repmat(mean(fixation_mu), ...
                       [size(trials_data_mu, 1) 1 1]);
baseline_beta = repmat(mean(fixation_beta), ...
                       [size(trials_data_beta, 1) 1 1]);

ERD_mu = log(trials_data_mu ./ baseline_mu);
ERD_beta = log(trials_data_beta ./ baseline_beta);


%% ERD/ERS visualization

figure;
t = linspace(0, min_trial_dur/sample_rate, min_trial_dur);
chan_selected = 9; 
colors = {'r', 'g'};
class_ID_selected = [773 771];
nclasses = length(class_ID_selected);

subplot(1, 2, 1);
h_mu = nan(nclasses, 1);

for i = 1:nclasses
    hold on;

    h_mu(i) = plot(t, mean(ERD_mu(:, chan_selected, trial_cue == class_ID_selected(i)), 3), colors{i});
    plot(t, mean(ERD_mu(:, chan_selected, trial_cue == class_ID_selected(i)), 3) + std(ERD_mu(:, chan_selected, trial_cue == class_ID_selected(i))/sqrt(sum(trial_cue == class_ID(i))), [], 3), [colors{i} ':']);
    plot(t, mean(ERD_mu(:, chan_selected, trial_cue == class_ID_selected(i)), 3) - std(ERD_mu(:, chan_selected, trial_cue == class_ID_selected(i))/sqrt(sum(trial_cue == class_ID(i))), [], 3), [colors{i} ':']);

    hold off;

end

title(['ERD in mu band | Mean +/- SE | channel ' chan_label{chan_selected}]);
xlabel('Time [s]');
ylabel('[%]');
line([1 1],get(gca,'YLim'),'Color',[0 0 0])
grid on;
legend(h_mu, 'both hands', 'both feet');

subplot(1, 2, 2);
h_beta = nan(nclasses, 1);
for i = 1:nclasses
    hold on;
    h_beta(i) = plot(t, mean(ERD_beta(:, chan_selected, trial_cue == class_ID_selected(i)), 3), colors{i});
    plot(t, mean(ERD_beta(:, chan_selected, trial_cue == class_ID_selected(i)), 3) + std(ERD_beta(:, chan_selected, trial_cue == class_ID_selected(i))/sqrt(ntrials/2), [], 3), [colors{i} ':']);
    plot(t, mean(ERD_beta(:, chan_selected, trial_cue == class_ID_selected(i)), 3) - std(ERD_beta(:, chan_selected, trial_cue == class_ID_selected(i))/sqrt(ntrials/2), [], 3), [colors{i} ':']);
    hold off;
end
title(['ERD in beta band | Mean +/- SE | channel ' chan_label{chan_selected}]);
xlabel('Time [s]');
ylabel('[%]');
line([1 1],get(gca,'YLim'),'Color',[0 0 0])
grid on;
legend(h_beta, 'both hands', 'both feet');

sgtitle(['ERD/ERS - Channel ' chan_label{chan_selected} ' for subject ' sub_id{1}(1:3)]);

%% Visualization with topoplot

fig2 = figure;
load('chanlocs16.mat');
period = [1 1.5] * sample_rate;


for i = 1:nclasses
    subplot(2, 2, i);
    data = mean(mean(ERD_mu(period(1):period(2), :, trial_cue == class_ID_selected(i)), 3), 1);
    topoplot(squeeze(data), chanlocs16, 'headrad', 'rim', 'maplimits', [-0.6 0.6]);
    axis image;
    title(['ERD/ERS in mu for ' class_label{i}])
end

for i = 1:nclasses
    subplot(2, 2, 2+i);
    data = mean(mean(ERD_beta(period(1):period(2), :, trial_cue == class_ID_selected(i)), 3), 1);
    topoplot(squeeze(data), chanlocs16, 'headrad', 'rim', 'maplimits', [-1.5 1.5]);
    axis image;
    title(['ERD/ERS in beta for ' class_label{i}])
end

sgtitle(['ERD/ERS (topoplot) for subject ' sub_id{1}(1:3)]);

drawnow;