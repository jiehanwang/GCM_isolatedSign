%%
clear all;
clc;
addpath(genpath('C:\UserData\iCode\GitHub\libsvm\matlab'));

% The class symbol list
names = importdata('.\input\protocol_370_Daily.txt');
%% Settings
nDim = 1162;
nSub = 10;              % The size of subspace
frameMode = 'allFrame'; %'allFrame';  keyPart
featureType = 'GREY';
nSample = 5;
nInsert = nSub;           % is the nFrame less than nInsert, enlarge it. 
nClass = length(names);
%% Readin the data
% Readin the data from original txt files.
routePath = 'C:\UserData\iData\Outputs\ftdcgrs_whj_output\dim1162_grey_allFrame_370sign_D\';

% The data path
path{1}  = [routePath 'P50\'];
path{2}  = [routePath 'P51\'];
path{3}  = [routePath 'P52\'];
path{4}  = [routePath 'P53\'];
path{5}  = [routePath 'P54\'];


% Readin the data
for i=1:nClass
    fprintf('Readin Data: %d / %d\n', i, nClass);
    for s=1:nSample
        dataName = [path{s}(end-3:end-1) '_' names{i}(2:5)];
        fileName = sprintf('%s%s.txt', path{s}, dataName);
%         dataName = names{i};
%         fileName = sprintf('%s%s.txt', path{s}, dataName);
        
        % Readin the i_th class, s_th sample
        D_temp = importdata(fileName, ' ', 1);
        %data_norm = (insertFrame(D_temp.data,nInsert))';
        data_norm = (D_temp.data)';     %这里可以用上面语句进行插值
        sub_data = construct_subspace(data_norm, nSub);
        D(i).Sample{s} = sub_data;
    end
end
% save('dim658sub10_lbp_keyPart_370sign_5group_D.mat','D');
% load D:\iData\Outputs\ftdcgrs_publishData_whj_output\dim334_CTskp_allFrame_2000sign_12group\matFiles\dim334sub10_CTskp_allFrame_2000sign_12group;
%%
nTest = 1;
accuracy = zeros(nSample, 1);
for testSample = 1:nSample
    trainNum = nSample - nTest;
    training_label = zeros(trainNum*length(names), 1);
    test_label = zeros(nTest*length(names), 1);
    training_data = [];
    test_data = [];
    model_precomputed = [];
    trainN = 1;
    testN = 1;
    for i = 1 : nClass
        fprintf('Assign training and test: Sign %d\n', i);
        for s=1:nSample   
            if (s~=testSample) 
                training_data{trainN} = D(i).Sample{s};
                training_label(trainN) = str2double(names{i}(2:5));
                trainN  = trainN+1;
            else                      % For test
                test_data{testN} = D(i).Sample{s};
                test_label(testN) = str2double(names{i}(2:5));
                testN = testN+1;
            end
        end
    end
    trainN = trainN-1;
    testN = testN-1;
    %% SVM trainging and test
    % Trainging
    TrainKernel = kernel(training_data,[],testSample);
    TTrainKernel = [(1:length(names)*trainNum)',TrainKernel];
    model_precomputed = svmtrain(training_label, TTrainKernel, '-t 4');

    % Test
    ValKernel = kernel(training_data,test_data,testSample);
    VValKernel = [(1:length(names)*nTest)',ValKernel'];
    [predict_label, accuracy_p, dec_values] = svmpredict(test_label, VValKernel, model_precomputed);
    accuracy(testSample) = accuracy_p(1);
    
    % Result saving.
%     T = fix(clock);
    prefix = sprintf('result\\GCM_%s_%d_%d_%s_%dsign_%dg', ...
        featureType, nDim,nSub, frameMode, nClass, nSample);
    fileName_middleResult = [prefix num2str(testSample+49) '.mat'];
    save(fileName_middleResult, 'VValKernel', 'accuracy_p', 'test_label', 'model_precomputed', '-v7.3');

    fileName_result = [prefix '.txt'];
    fid = fopen(fileName_result,'at+');
    fprintf(fid,'P%02d\t%f\n',testSample+49, accuracy_p(1));
    fclose(fid);
end