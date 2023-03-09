clearvars -except sub_id subjects erd_ers_gdf erd_ers_psd psd_computing model_training model_testing subjects_selected class_model;

%% Import PSD data (offline runs only)

% Data information

datapath = 'micontinuous/';
chan_label = {'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C3', 'C1', 'Cz', 'C2', 'C4', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4'};
class_ID = [771 773 783];   
class_label = {'both hands', 'both feet', 'rest'};
nclasses = length(class_ID);
mod_ID = [0 1];
mod_label = {'offline', 'online'};

disp(['Model training for subject ' sub_id{1}]);

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

%% Fisher score

selfreqs = 4:2:48;

[freqs, idfreqs] = intersect(freqs, selfreqs);
nfreqs = size(freqs,1);

log_psd = log(all_psd(:,idfreqs,:));

unique_runs = unique(runs);
fisher_score = nan(nfreqs, nchans, size(unique_runs,1));

for j = 1:size(unique_runs,1)

    run_index = (runs == unique_runs(j));

    mu = nan(nfreqs, nchans, 2);
    sigma = nan(nfreqs, nchans, 2);

    for i = 1:(nclasses-1)
        index = (run_index & cue == class_ID(i));
        mu(:, :, i) = squeeze(mean(log_psd(index, :, :)));
        sigma(:, :, i) = squeeze(std(log_psd(index, :, :)));
    end

    fisher_score(:, :, j) = abs(mu(:, :, 2) - mu(:, :, 1)) ./ sqrt((sigma(:, :, 1) .^ 2 + sigma(:, :, 2) .^2));
end

% Visualizing fisher score

limits = [];
handles = nan(length(unique_runs), 1);
fig1 = figure;

for i = 1:length(unique_runs)
    subplot(1, length(unique_runs), i);
    imagesc(fisher_score(:, :, unique_runs(i))');
    
    axis square;
    set(gca, 'XTick', 1:nfreqs);
    set(gca, 'XTickLabel', selfreqs);
    set(gca, 'YTick', 1:nchans);
    set(gca, 'YTickLabel', chan_label);
    xtickangle(-90);

    title(['Calibration run ' num2str(unique_runs(i))]);

    limits = cat(2, limits, get(gca, 'CLim'));
    handles(unique_runs(i)) = gca;
end

set(handles, 'clim', [min(min(limits)) max(max(limits))]);

sgtitle(['Subject ' sub_id{1}(1:3) ': Fisher score']);

%% Features selection

disp('Selecting features ...');

nfeatures = 3;

% Select most important features
fisher_score_avg = mean(fisher_score, 3);

maxk_fisher_score_avg = maxk(fisher_score_avg(:), nfeatures);
[f, c] = find(fisher_score_avg >= maxk_fisher_score_avg(nfeatures));

selected_chans = chan_label(c);
selected_freqs = selfreqs(f);

[~, selected_chans_ID] = ismember(selected_chans, chan_label);
[~, selected_freqs_ID] = ismember(selected_freqs, freqs);

fts = nan(nwindows, nfeatures);

for i = 1:nfeatures
    freq = selected_freqs_ID(i);
    chan = selected_chans_ID(i);
    fts(:, i) = log_psd(:, freq, chan);
end

%% Classifier

disp('Training classifier with offline data...');
LabelIdx = (cue == 771 | cue == 773 | cue == 783);

if class_model == 1
    model = fitcdiscr(fts(LabelIdx, :), cue(LabelIdx), 'DiscrimType', 'quadratic');
else
    model = fitcsvm(fts(LabelIdx, :), cue(LabelIdx), "KernelFunction", "rbf");
end

[Gk, pp] = predict(model, fts);

ss_acc = 100 * sum(Gk(LabelIdx) == cue(LabelIdx)) ./ length(Gk(LabelIdx));
classes = [773 771];
classes_labels = {'both hands', 'both feet'};
ss_cl_acc = nan(size(classes,2), 1);

for i = 1:size(classes,2)
    index = (cue == classes(i));
    ss_cl_acc(i) = 100 * sum(Gk(index) == cue(index)) ./ length(Gk(index));
end

disp(['Single sample accuracy on training: ' num2str(ss_acc)]);
disp(['Single sample accuracy on training for 771, 773: ' num2str(ss_cl_acc(1)) ', ' num2str(ss_cl_acc(2))]);

disp('Saving classifier ...');
filename = [sub_id{1} '_classifier.mat'];
save(filename, 'model', 'selected_chans_ID', 'selected_freqs_ID');

% Visualizing classifier

disp('Visualizing classifier ...')
data = fts(LabelIdx, :);
figure;

if class_model == 1
    h1 = gscatter(fts(LabelIdx, 1), fts(LabelIdx, 2), cue(LabelIdx), 'kb', 'ov^', [], 'off');
    
    hold on;

    K = model.Coeffs(1, 2).Const;
    L = model.Coeffs(1, 2).Linear;
    Q = model.Coeffs(1, 2).Quadratic;
    f = @(x1, x2) K + L(1)*x1 + L(2)*x2 + Q(1, 1)*x1.^2 + ...
        (Q(1, 2) + Q(2, 1))*x1.*x2 + Q(2, 2)*x2.^2;
    
    h2 = fimplicit(f);
    h2.Color = 'r';
    h2.LineWidth = 2;
    h2.DisplayName = 'Boundary between both hands & both feet';
    legend('both feet', 'both hands', 'Boundary');
    hold off;
else
    scatter3(fts(cue==771, 1), fts(cue==771, 2), fts(cue==771, 3), "green", "o");
    hold on;
    scatter3(fts(cue==773, 1), fts(cue==773, 2), fts(cue==773, 3), "red", "^");
    
    plot3(model.SupportVectors(:,1),model.SupportVectors(:,2), model.SupportVectors(:,3),'ko','MarkerSize',10)
    legend('both feet','both hands','Support Vector');
    hold off;
end

grid on;
xlabel([selected_chans{1} '@' num2str(selected_freqs(1)) 'Hz']);
ylabel([selected_chans{2} '@' num2str(selected_freqs(2)) 'Hz']);
zlabel([selected_chans{3} '@' num2str(selected_freqs(3)) 'Hz']);

axis auto;

title(['Subject ' sub_id{1}(1:3) ': Classifier'])

% Accuracy on training set

figure;
bar([ss_acc; ss_cl_acc]);
grid on;
set(gca, 'XTickLabel', {'overall', classes_labels{1} classes_labels{2}});
ylim([0 100]);
ylabel('accuracy [%]');
title(['Subject ' sub_id{1}(1:3) ': Accuracy on train set']);

drawnow;


%% Classifier (LSTM)
% Uncomment the following code to train a Long Short-Term Memory model
%
% disp('Training classifier (LSTM) with offline data ...');
% 
% X = cell(1, ntrials);
% Y = cell(ntrials, 1);
% 
% for i = 1:ntrials
%     X{i} = fts(TrialStart(i):TrialStop(i), :)';
%     Y{i, 1} = CueTYP(i);
% end
% 
% validation_split = floor(ntrials * 0.75);
% XTrain = X(1:validation_split);
% YTrain = Y(1:validation_split,1);
% XValid = X(validation_split+1:ntrials);
% YValid = Y(validation_split+1:ntrials, 1);
% 
% YTrain = cellfun(@num2str,YTrain,'uni',0);
% YTrain = categorical(YTrain);
% YValid = cellfun(@num2str,YValid,'uni',0);
% YValid = categorical(YValid);
% 
% inputSize = nfeatures;
% numHiddenUnits = 50;
% numClasses = size(classes,2);
% 
% layers = [ ...
%     sequenceInputLayer(inputSize)
%     bilstmLayer(numHiddenUnits,'OutputMode','last')
%     fullyConnectedLayer(numClasses)
%     softmaxLayer
%     classificationLayer];
% 
% maxEpochs = 100;
% miniBatchSize = 30;
% 
% options = trainingOptions('adam', ...
%     'ExecutionEnvironment','gpu', ...
%     'GradientThreshold', 1, ...
%     'MaxEpochs',maxEpochs, ...
%     'MiniBatchSize',miniBatchSize, ...
%     'SequenceLength','longest', ...
%     'Shuffle','never', ...
%     'Verbose',0, ...
%     'Plots','none');
% 
% net = trainNetwork(XTrain, YTrain, layers,options);
% 
% [YPred, pp] = classify(net,XValid, ...
%     'MiniBatchSize',miniBatchSize, ...
%     'SequenceLength','longest');
% 
% ss_acc = 100 * sum(YPred == YValid)./numel(YValid);
% 
% disp('Saving classifier (LSTM) ...');
% filename = [sub_id{1} '_LSTM.mat'];
% save(filename, 'net', 'selected_chans_ID', 'selected_freqs_ID', 'miniBatchSize');
% 
% classes = [773 771];
% classes_labels = {'both hands', 'both feet'};
% ss_cl_acc = nan(size(classes,2), 1);
% 
% cell_classes = num2cell(classes);
% cat_classes = categorical(cellfun(@num2str,cell_classes,'uni',0));
% 
% for i = 1:size(cat_classes,2)
%     index = (YValid == cat_classes(i));
%     ss_cl_acc(i) = 100 * sum(YPred(index) == YValid(index)) ./ length(YPred(index));
% end
% 
% % Accuracy on training set
% 
% figure;
% bar([ss_acc; ss_cl_acc]);
% grid on;
% set(gca, 'XTickLabel', {'overall', classes_labels{1} classes_labels{2}});
% ylim([0 100]);
% ylabel('accuracy [%]');
% title(['Subject ' sub_id{1}(1:3) ' accuracy on train set (LSTM)']);
% 
% drawnow;