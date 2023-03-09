clearvars -except sub_id subjects erd_ers_gdf erd_ers_psd psd_computing model_training model_testing subjects_selected class_model;

%% Import PSD data (online runs only)

% Data information

datapath = 'micontinuous/';
chan_label = {'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C3', 'C1', 'Cz', 'C2', 'C4', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4'};
class_ID = [771 773 783];   
class_label = {'both hands', 'both feet', 'rest'};
nclasses = length(class_ID);
mod_ID = [0 1];
mod_label = {'offline', 'online'};

disp(['Model testing for subject ' sub_id{1}]);

files = dir(fullfile([datapath '/' sub_id{1}], '*.mat'));
nfiles = size(files,1);

all_psd = [];
runs = [];
TYP = [];
DUR = [];
POS = [];

for i = 1:nfiles
    filename = strcat(datapath, sub_id{1}, '/', files(i).name);

    if(contains(filename,'online'))
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

nwindows = size(all_psd, 1);
nchans = size(all_psd, 3);

FeedbackPOS = POS(TYP == 781);
FeedbackDUR = DUR(TYP == 781);

CuePOS = POS(TYP == 771 | TYP == 773 | TYP == 783);
CueDUR = DUR(TYP == 771 | TYP == 773 | TYP == 783);
CueTYP = TYP(TYP == 771 | TYP == 773 | TYP == 783);

% Trial information extraction

ntrials = length(FeedbackPOS);    
cue = zeros(nwindows, 1);
trials = zeros(nwindows, 1);
TrialStart = nan(ntrials, 1);
TrialStop  = nan(ntrials, 1);

for i = 1:ntrials
    current_start = CuePOS(i);
    current_stop  = FeedbackPOS(i) + FeedbackDUR(i) - 1;
    cue(current_start:current_stop) = CueTYP(i);
    trials(current_start:current_stop) = i;
    
    TrialStart(i) = current_start;
    TrialStop(i)  = current_stop;
end

%% Applying log to PSD

selfreqs = 4:2:48;

[freqs, idfreqs] = intersect(freqs, selfreqs);
nfreqs = size(freqs,1);

log_psd = log(all_psd(:,idfreqs,:));

%% Loading classifier

classifier = load([sub_id{1} '_classifier.mat']);

model = classifier.model;
selected_chans_ID = classifier.selected_chans_ID;
selected_freqs_ID = classifier.selected_freqs_ID;

nfeatures = length(selected_chans_ID);
fts = nan(nwindows, nfeatures);

for i = 1:nfeatures
    freq = selected_freqs_ID(i);
    chan = selected_chans_ID(i);
    fts(:, i) = log_psd(:, freq, chan);
end

%% Classification on online data

disp('Evaluate classifier');
LabelIdx = (cue == 771 | cue == 773);

[Gk, pp] = predict(model, fts);

ss_acc = 100*sum(Gk(LabelIdx) == cue(LabelIdx)) ./ length(Gk(LabelIdx));
classes = [773 771];
classes_labels = {'both hands', 'both feet'};
ss_cl_acc = nan(size(classes,2), 1);

for i = 1:size(classes,2)
    index = (cue == classes(i));
    ss_cl_acc(i) = 100 * sum(Gk(index) == cue(index)) ./ length(Gk(index));
end

disp(['Single sample accuracy on test data: ' num2str(ss_acc)]);
disp(['Single sample accuracy on test data for 771, 773: ' num2str(ss_cl_acc(1)) ', ' num2str(ss_cl_acc(2))]);

figure;
bar([ss_acc; ss_cl_acc]);
grid on;
set(gca, 'XTickLabel', {'overall', classes_labels{1} classes_labels{2}});
ylim([0 100]);
ylabel('accuracy [%]');
title(['Subject ' sub_id{1}(1:3) ': Accuracy on test set']);

drawnow;

%% Classification on online data (LSTM)
% Uncomment the following code to train a Long Short-Term Memory model
% lstm = load([sub_id{1} '_LSTM.mat']);
% net = lstm.net;
% miniBatchSize = lstm.miniBatchSize;
% 
% disp('Evaluate classifier');
% LabelIdx = (cue == 771 | cue == 773);
% 
% XTest = cell(1, ntrials);
% YTest = cell(ntrials, 1);
% 
% for i = 1:ntrials
%     XTest{i} = fts(TrialStart(i):TrialStop(i), :)';
%     YTest{i, 1} = CueTYP(i);
% end
% 
% YTest = cellfun(@num2str,YTest,'uni',0);
% YTest = categorical(YTest);
% 
% YPred = classify(net,XTest, ...
%     'MiniBatchSize',miniBatchSize, ...
%     'SequenceLength','longest');
% 
% ss_acc = 100 * sum(YPred == YTest)./numel(YTest);
% 
% cell_classes = num2cell(classes);
% cat_classes = categorical(cellfun(@num2str,cell_classes,'uni',0));
% 
% for i = 1:size(cat_classes,2)
%     index = (YTest == cat_classes(i));
%     ss_cl_acc(i) = 100 * sum(YPred(index) == YTest(index)) ./ length(YPred(index));
% end
% 
% figure;
% bar([ss_acc; ss_cl_acc]);
% grid on;
% set(gca, 'XTickLabel', {'overall', classes_labels{1} classes_labels{2}});
% ylim([0 100]);
% ylabel('accuracy [%]');
% title(['Subject ' sub_id{1}(1:3) ' accuracy on test set (LSTM)']);
% 
% drawnow;

%% Evidence accumulation framework 1

if class_model == 0
    model = fitPosterior(model);
end

trial_start_online = TrialStart;
trial_stop_online = TrialStop;
ntrials = size(trial_start_online,1);
a = 0.1;
trial_correct = 0;
upper_treshold = 0.8;
lower_treshold = 0.2;
figure;
hold on;

for i = 1:ntrials
    D = 0.5;
    all_D = [];
    
    start = trial_start_online(i);
    stop = trial_stop_online(i);
    output = class_ID(3); %default class: rest condition

    j = start;

    while j < stop

        [label, pp]= predict(model,fts(j,:));

        % Posterior probability considered: both feet (771)
        D = (1 - a) * D + a * pp(1); 
        j = j + 1;
        
        if mod(i,10) == 0
            all_D = cat(1,all_D,D);
        end

        % High values of D -> both feet
        % Low values of D -> both hands

        if D > upper_treshold && output == class_ID(3)
            output = class_ID(1);
            d_stop = j;
        elseif D < lower_treshold && output == class_ID(3)
            output = class_ID(2);
            d_stop = j;
        end

    end

    if mod(i, 10) == 0
        subplot(3, 4, i / 10);
        color = '-k';

        switch(output)
            case 771
                color = '-g';
            case 773
                color = '-r';
        end
        
        plot(all_D, color, 'LineWidth', 1);
        title(['Cue: ' num2str(cue(start)) ', predicted: ' num2str(output)]);
        yline(upper_treshold, '-k', 'Both feet');
        yline(lower_treshold, '-k', 'Both hands');
        axis([0 inf 0 1]);
        drawnow;
    end

    if output == cue(j)
        trial_correct = trial_correct + 1;
    end

end

sgtitle(['Evidence accumulation framework for subject ' sub_id{1}(1:3)]);

avg_trial_acc = 100 * trial_correct / ntrials;
disp(strcat('Average accuracy on trials: ', string(avg_trial_acc)));

%% Evidence accumulation framework 2

trial_start_online = TrialStart;
trial_stop_online = TrialStop;
ntrials = size(trial_start_online,1);
trial_correct = 0;
entropy_treshold = 0.4;
figure;

for i = 1:ntrials
    D = 0.5;
    all_D = [];
    
    start = trial_start_online(i);
    stop = trial_stop_online(i);
    output = class_ID(3); %default class: rest condition

    j = start;

    while j < stop

        [label, pp]= predict(model,fts(j,:));

        entropy = 0;

        for k=1:size(pp,2)
            entropy = entropy - pp(k) * log2(pp(k));
        end

        if entropy < entropy_treshold
            % Posterior probability considered: both feet (771)
            D = (1 - a) * D + a * pp(1);
        end

        j = j + 1;
        
        if mod(i,10) == 0
            all_D = cat(1,all_D,D);
        end

        % High values of D -> both feet
        % Low values of D -> both hands

        if D > upper_treshold && output == class_ID(3)
         output = class_ID(1);
        elseif D < lower_treshold && output == class_ID(3)
         output = class_ID(2);
        end

    end

    if mod(i, 10) == 0
        subplot(3, 4, i / 10);
        color = '-k';

        switch(output)
            case 771
                color = '-g';
            case 773
                color = '-r';
        end
        
        plot(all_D, color, 'LineWidth', 1);
        title(['Cue: ' num2str(cue(start)) ', predicted: ' num2str(output)]);
        yline(upper_treshold, '-k', 'Both feet');
        yline(lower_treshold, '-k', 'Both hands');
        axis([0 inf 0 1]);
        drawnow;
    end

    if output == cue(j)
    trial_correct = trial_correct + 1;
    end

end

sgtitle(['Enhanced evidence accumulation framework for subject ' sub_id{1}(1:3)]);

avg_trial_acc = 100 * trial_correct / ntrials;
disp(strcat('Average accuracy on trials (2): ', string(avg_trial_acc)));

