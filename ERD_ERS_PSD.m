clearvars -except sub_id subjects erd_ers_gdf erd_ers_psd psd_computing model_training model_testing subjects_selected class_model;

%% Import PSD data (offline runs only)

% Data information

datapath = 'micontinuous/';
chan_label = {'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C3', 'C1', 'Cz', 'C2', 'C4', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4'};
mod_ID = [0 1];
mod_label = {'offline', 'online'};

disp(['ERD/ERS on PSD for subject ' sub_id{1}]);

files = dir(fullfile([datapath '/' sub_id{1}], '*.mat'));
nfiles = size(files,1);

all_psd = [];
runs = [];
TYP = [];
DUR = [];
POS = [];

for i = 1:nfiles

    filename = strcat(datapath, sub_id{1}, '/', files(i).name);

    if(contains(filename,'offline'))
        current_data = load(filename);
        
        psd = current_data.psd;

        current_run = i * ones(size(psd, 1), 1);
        runs = cat(1, runs, current_run);
    
        win_events = current_data.win_events;

        TYP = cat(1, TYP, win_events.TYP);
        DUR = cat(1, DUR, win_events.DUR);
        POS = cat(1, POS, win_events.POS + size(all_psd, 1));

        all_psd = cat(1,all_psd,psd);
        
        freqs = current_data.freqs;
        sample_rate = current_data.sample_rate;
        info = current_data.info;
    end
end

% Extract events information

nwindows = size(all_psd, 1);
nfreqs = size(all_psd, 2);
nchannels = size(all_psd, 3);

FeedbackPOS = POS(TYP == 781);
FeedbackDUR = DUR(TYP == 781);

CuePOS = POS(TYP == 771 | TYP == 773 | TYP == 783);
CueDUR = DUR(TYP == 771 | TYP == 773 | TYP == 783);
CueTYP = TYP(TYP == 771 | TYP == 773 | TYP == 783);

FixPOS = POS(TYP == 786);
FixDUR = DUR(TYP == 786);
FixTYP = TYP(TYP == 786);

% Trial information extraction

ntrials = length(FeedbackPOS);    
cue = zeros(nwindows, 1);
trials = zeros(nwindows, 1);
TrialStart = nan(ntrials, 1);
TrialStop  = nan(ntrials, 1);
FixStart = nan(ntrials, 1);
FixStop  = nan(ntrials, 1);

for i = 1:ntrials

    current_start = CuePOS(i);
    current_stop  = FeedbackPOS(i) + FeedbackDUR(i) - 1;
    cue(current_start:current_stop) = CueTYP(i);
    trials(current_start:current_stop) = i;
    
    TrialStart(i) = current_start;
    TrialStop(i) = current_stop;
    FixStart(i) = FixPOS(i);
    FixStop(i) = FixPOS(i) + FixDUR(i) - 1;
end

% Extracting data for each trial

min_trial_dur = min(TrialStop - TrialStart);
trials_data = nan(min_trial_dur, nfreqs, nchannels, ntrials);
trial_cue = zeros(ntrials, 1);
trial_run = zeros(ntrials, 1);

for i = 1:ntrials

    trials_data(:, :, :, i) = all_psd(TrialStart(i):TrialStart(i) + min_trial_dur - 1, :, :);
    trial_cue(i) = unique(cue(TrialStart(i):TrialStart(i) + min_trial_dur - 1));
    trial_run(i) = unique(runs(TrialStart(i):TrialStart(i) + min_trial_dur - 1));
end

%% Baseline extraction using fixation period as reference

min_fix_dur = min(FixStop - FixStart);
fixation = nan(min_fix_dur, nfreqs, nchannels, ntrials);

for i = 1:ntrials

    fixation(:, :, :, i)   = all_psd(FixStart(i):FixStart(i) + min_fix_dur - 1, :, :);

end

%% ERD/ERS

disp('Computing ERD/ERS on PSD data');

% Average and replicate the value of the baseline
baseline = repmat(mean(fixation), [size(trials_data, 1) 1 1 1]);
ERD = log(trials_data ./ baseline);

% ERD/ERS visualization

figure;
hold on;
t = linspace(0, min_trial_dur*info.wshift, min_trial_dur);
chan_selected = [7 9 11]; 
class_ID_selected = [773 771];
class_label_selected = {'Both hands', 'Both feet'};
nclasses = length(class_ID_selected);

chandles = [];

for i = 1:nclasses

    limits = nan(2, length(chan_selected));

    for j = 1:length(chan_selected)
        subplot(2, 3, (i - 1)*length(chan_selected) + j);
        data = mean(ERD(:, :, chan_selected(j), ...
                    trial_cue == class_ID_selected(i)), 4);
        imagesc(t, freqs, data');
        set(gca,'YDir','normal');
        limits(:, j) = get(gca, 'CLim');
        chandles = cat(1, chandles, gca);
        colormap(hot);
        colorbar;
        title(['Channel ' chan_label{chan_selected(j)} ' | ' ...
                          class_label_selected{i}]);
        xlabel('Time [s]');
        ylabel('Frequency [Hz]');
        line([1 1],get(gca,'YLim'),'Color',[0 0 0])
    end
 
end

set(chandles, 'CLim', [min(min(limits)) max(max(limits))]);
sgtitle(['ERD/ERS (PSD) for subject ' sub_id{1}(1:3)]);
hold off;
drawnow;
