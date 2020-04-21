% https://www.mathworks.com/help/matlab/ref/save.html
% save your mat file with v7.3
% To view or set the default version for MAT-files, go to the Home tab and in the Environment section, click  Preferences. 
% Select MATLAB > General > MAT-Files and then choose a MAT-file save format option.

clear,clc;

%% KoNViD-1k
data_path = '/media/ldq/Research/Data/KoNViD-1k/KoNViD_1k_attributes.csv';
data = readtable(data_path);
video_names = data.file_name; % video names
scores = data.MOS; % subjective scores
clear data_path data

height = 540; % video height
width = 960; % video width
max_len = 240; % maximum video length in the dataset
video_format = 'RGB'; % video format
ref_ids = [1:length(scores)]'; % video content ids
% `random` train-val-test split index, 1000 runs
index = cell2mat(arrayfun(@(i)randperm(length(scores)), ...
    1:1000,'UniformOutput', false)'); 
save('KoNViD-1kinfo','-v7.3')

%% CVD2014
data_path = '/media/ldq/Research/Data/CVD2014/CVD2014_ratings/Realignment_MOS.csv';
data = readtable(data_path);
video_names = arrayfun(@(i) ['Test' data.File_name{i}(6) '/' ...
    data.Content{i} '/' data.File_name{i} '.avi'], 1:234, ...
    'UniformOutput', false)';  % video names, remove '', add dir
scores = arrayfun(@(i) str2double(data.RealignmentMOS{i})/100, 1:234)'; % subjective scores
clear data_path data

height = [720 480];
width = [1280 640];
max_len = 830;
video_format = 'RGB';
ref_ids = [1:length(scores)]';
% `random` train-val-test split index, 1000 runs
index = cell2mat(arrayfun(@(i)randperm(length(scores)), ...
    1:1000,'UniformOutput', false)'); 
save('CVD2014info','-v7.3')
% LIVE-Qualcomm
data_path = '/media/ldq/Others/Data/12.LIVE-Qualcomm Mobile In-Capture Video Quality Database/qualcommSubjectiveData.mat';
data = load(data_path);
scores = data.qualcommSubjectiveData.unBiasedMOS; % subjective scores
video_names = data.qualcommVideoData;
video_names = arrayfun(@(i) [video_names.distortionNames{video_names.distortionType(i)} ...
    '/' video_names.vidNames{i}], 1:length(scores), ...
    'UniformOutput', false)'; % video names
clear data_path data

height = 1080;
width = 1920;
max_len = 526;
video_format = 'YUV420';
ref_ids = [1:length(scores)]';
% `random` train-val-test split index, 1000 runs
index = cell2mat(arrayfun(@(i)randperm(length(scores)), ...
    1:1000,'UniformOutput', false)'); 
save('LIVE-Qualcomminfo','-v7.3')
