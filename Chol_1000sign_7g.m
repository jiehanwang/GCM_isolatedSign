%%
clear all;
clc;
addpath(genpath('D:\iCode\GitHub\libsvm\matlab'));

% The class symbol list
names = importdata('.\input\protocol_1000_7groups.txt');
%% Settings
nDim = 334;
frameMode = 'allFrame';
nSample = 7;
nClass = length(names);
%% Readin the data
% Readin the data from original txt files.
routePath = 'D:\iData\Outputs\ftdcgrs_whj_output\dim334_CTskp_allFrame_1000sign_7group\';

% The data path
path{1}  = [routePath 'test_10\'];
path{2}  = [routePath 'test_11\'];
path{3}  = [routePath 'test_14\'];
path{4}  = [routePath 'test_15\'];
path{5}  = [routePath 'test_17\'];
path{6}  = [routePath 'test_19\'];
path{7}  = [routePath 'test_21\'];

% Readin the data
for i=1:nClass
    fprintf('Readin Data: %d / %d\n', i, nClass);
    for s=1:nSample
        dataName = names{i};
        fileName = sprintf('%s%s.txt', path{s}, dataName);
        
        % Readin the i_th class, s_th sample
        D_temp = importdata(fileName, ' ', 1);
        data_norm = (D_temp.data)';
        Cov = CovarianceMatrx(data_norm);
        L = chol(Cov,'lower');
        D(i).Sample{s} = L;
    end
end
save('dim334Chol_CTskp_allFrame_1000sign_7group.mat','D');
% load D:\iData\Outputs\ftdcgrs_whj_output\dim334_CTskp_allFrame_1000sign_7group\matFiles\dim334sub10_CTskp_allFrame_1000sign_7group
%% Assign traing and test data
testSample = 1;
nTest = 1;

trainNum = nSample - nTest;
training_label = zeros(trainNum*length(names), 1);
test_label = zeros(nTest*length(names), 1);
training_data = [];
test_data = [];
trainN = 1;
testN = 1;
for i = 1 : nClass
    fprintf('Assign training and test: Sign %d\n', i);

    % Subspace extraction
    for s=1:nSample   
        if (s~=testSample) % || (s==testSample && i<501)     % For training  “第9组以前训练，以后测试”
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

% Test 四组一起测试
ValKernel = kernel(training_data,test_data,testSample);
VValKernel = [(1:length(names)*nTest)',ValKernel'];
[predict_label, accuracy_p, dec_values] = svmpredict(test_label, VValKernel, model_precomputed);

% % Result saving.
% prefix = sprintf('result\\GCM_%dto%d_%s_%dsign_%dg',nDim, nSub,frameMode, nClass, nSample);
% fileName_middleResult = [prefix num2str(testSample) '.mat'];
% % save(fileName_middleResult, 'test_data', 'training_data', 'TTrainKernel'...
% %     ,'VValKernel', 'accuracy', 'test_label', 'training_label', 'model_precomputed', '-v7.3');
% save(fileName_middleResult, 'VValKernel', 'accuracy', 'test_label', 'model_precomputed', '-v7.3');
% % fprintf(fid,'P%d %f\t\n',testSample, accuracy(1,1));
% 
% fileName_result = [prefix '.txt'];
% fid = fopen(fileName_result,'at+');
% for i=1:nTest
%     fprintf(fid,'P%02d\t%f\n',i+ID_testBegin-1, accuracy(i));
% end
% fclose(fid);