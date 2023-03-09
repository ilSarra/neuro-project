clearvars; clc;

datapath = 'micontinuous/';
files_in_datapath = dir(datapath);
folders_in_datapath = files_in_datapath([files_in_datapath.isdir]);
subjects = {folders_in_datapath(3:end).name};   % start from 3 to skip . and ..

[erd_ers_gdf, erd_ers_psd, psd_computing, model_training, model_testing, subjects_selected, class_model] = GUI(subjects);

%% Processing

for sub_id = subjects_selected

    % Visualizing ERD/ERS and topoplot on GDF data (offline runs)
    if erd_ers_gdf
        run("ERD_ERS_GDF.m");
    end

    % Computing PSD for all runs
    if psd_computing
        run("PSD_computing.m");
    end
    
    % Visualizing ERD/ERS on PSD data (offline runs)
    if erd_ers_psd
        run("ERD_ERS_PSD.m");
    end
    
    % Training model on offline runs
    if model_training
        run("Model_training.m");
    end
    
    % Model evaluation
    if model_testing
        run("Model_testing.m");
    end
end

%% GUI definition

function [erd_ers_gdf, erd_ers_psd, psd_computing, model_training, model_testing, subjects_selected, class_model] = GUI(subjects)

    fig = uifigure("Name","BMI PROCESSING");
    cb_pnl = uipanel(fig,"Title","Select processing procedures","FontWeight","bold",Position = [10 200 250 200]);
    
    cb_width = 200;
    cb_height = 15;
    
    cb_ERD_ERS_GDF = uicheckbox(cb_pnl,'Text','ERD/ERS on GDF files',Position=[10 140 cb_width cb_height]);
    cb_PSD_computing = uicheckbox(cb_pnl,'Text','PSD computing on GDF files', Position=[10 110 cb_width cb_height]);
    cb_ERD_ERS_PSD = uicheckbox(cb_pnl,'Text','ERD/ERS on PSD files', Position=[10 80 cb_width cb_height]);
    cb_Model_training = uicheckbox(cb_pnl,'Text','Train model on PSD data', Position=[10 50 cb_width cb_height]);
    cb_Model_testing = uicheckbox(cb_pnl,'Text','Evaluate model', Position=[10 20 cb_width cb_height]);
    
    rb_pnl = uibuttongroup(fig,"Title","Select classification model","FontWeight","bold",'Position',[10 100 250 80]);
    rb_width = 250;
    rb_height = 15;
    rb_QDA = uiradiobutton(rb_pnl, 'Text', 'QDA', Position=[10 35 rb_width rb_height], Value=1);
    rb_SVM = uiradiobutton(rb_pnl, 'Text', 'SVM', Position=[10 15 rb_width rb_height], Value=0);

    bt_continue = uibutton(fig, ...
                           'ButtonPushedFcn',@(bt_continue,event) startProcessing(fig), ...
                           'Text','OK', ...
                           'FontWeight','bold', ...
                            Position = [450 10 80 30]);
    lb_pnl = uipanel(fig,"Title","Select subjects","FontWeight","bold",Position = [280 100 200 300]);
    lb_subjects = uilistbox(lb_pnl,'Multiselect','on',Items = subjects,Position = [0 0 200 280]); 
    
    uiwait(fig);

    function startProcessing(fig)

    msg = 'Do you want to start the processing?';
    selection = uiconfirm(fig,msg,'Confirmation');

    switch selection
        case 'OK'
            erd_ers_gdf = get(cb_ERD_ERS_GDF,'Value');
            erd_ers_psd = get(cb_ERD_ERS_PSD,'Value');
            psd_computing = get(cb_PSD_computing,'Value');
            model_training = get(cb_Model_training,'Value');
            model_testing = get(cb_Model_testing,'Value');
            subjects_selected = get(lb_subjects,'Value');
            class_model = get(rb_QDA,'Value');
            close(fig);
        case 'Cancel'
            return
    end

    end

end

