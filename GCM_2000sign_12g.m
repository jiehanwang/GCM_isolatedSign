%%
clear all;
clc;
addpath(genpath('D:\iCode\GitHub\libsvm\matlab'));

% The class symbol list
names = importdata('.\input\protocol_2000_12g.txt');
%% Settings
nDim = 658;
nSub = 15;              % The size of subspace
frameMode = 'allFrame';
nSample = 12;
nInsert = nSub;           % is the nFrame less than nInsert, enlarge it. 
nClass = length(names);
%% Readin the data
% Readin the data from original txt files.
routePath = 'D:\iData\Outputs\ftdcgrs_whj_NavigateGesture_output\dim658_CTskp_allFrame_2000sign_12group\';

% The data path
path{1}  = [routePath 'P01_1\'];
path{2}  = [routePath 'P01_2\'];
path{3}  = [routePath 'P02_1\'];
path{4}  = [routePath 'P02_2\'];
path{5}  = [routePath 'P03_1\'];
path{6}  = [routePath 'P03_2\'];
path{7}  = [routePath 'P04_1\'];
path{8}  = [routePath 'P04_2\'];
path{9}  = [routePath 'P05_1\'];
path{10}  = [routePath 'P06_1\'];
path{11}  = [routePath 'P07_1\'];
path{12}  = [routePath 'P08_1\'];

% Readin the data
D.Sample = cell(1, nSample);
D = repmat(D, [nClass, 1]);
covGeneTime_2000sign_12group = 0;
for i=1:nClass
    fprintf('Readin Data: %d / %d\n', i, nClass);
    for s=1:nSample
        dataName = [path{s}(end-5:end-1) '_' names{i}(2:5)];
        fileName = sprintf('%s%s.txt', path{s}, dataName);
        
        % Readin the i_th class, s_th sample
        D_temp = importdata(fileName, ' ', 1);
        %data_norm = (insertFrame(D_temp.data,nInsert))';
        data_norm = (D_temp.data)';     %这里可以用上面语句进行插值
        tic;
        sub_data = construct_subspace(data_norm, nSub);
        covGeneTime_2000sign_12group = covGeneTime_2000sign_12group + toc;
        D(i).Sample{s} = sub_data;
    end
end
save('dim658sub15_CTskp_allFrame_2000sign_12group.mat','D', 'covGeneTime_2000sign_12group', '-v7.3');
% load D:\iData\Outputs\ftdcgrs_publishData_whj_output\dim334_CTskp_allFrame_2000sign_12group\matFiles\dim334sub10_CTskp_allFrame_2000sign_12group;
%% Assign traing and test data
testSample = 9101112;
nTest = 4;
ID_testBegin = 9;

trainNum = nSample - nTest;
training_label = zeros(trainNum*length(names), 1);
test_label = zeros(nTest*length(names), 1);
training_data = cell(trainNum*length(names), 1);
test_data = cell(nTest*length(names), 1);
trainN = 1;
testN = 1;
for i = 1 : nClass
    fprintf('Assign training and test: Sign %d\n', i);

    % Subspace extraction
    for s=1:nSample   
        if s < ID_testBegin        % For training  “第9组以前训练，以后测试”
            training_data{trainN} = D(i).Sample{s};
            training_label(trainN) = str2double(names{i}(2:5));
            trainN  = trainN+1;
        else                       % For test
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
tic;
TrainKernel = kernel(training_data,[],testSample);
TTrainKernel = [(1:length(names)*trainNum)',TrainKernel];
model_precomputed = svmtrain(training_label, TTrainKernel, '-t 4');
trainTime = toc;

% Test 四组一起测试
tic;
ValKernel = kernel(training_data,test_data,testSample);
VValKernel = [(1:length(names)*nTest)',ValKernel'];
% [predict_label, accuracy_p, dec_values] = svmpredict(test_label, VValKernel, model_precomputed);
% 如果内存不够，用下面的逐一测试
for i=1:nClass*nTest
    fprintf('Testing...%d / %d\n', i, nClass*nTest)
    [predict_label(i,1), ~, ~] = svmpredict(test_label(i,:), VValKernel(i,:), model_precomputed, '-q');
end
testTime = toc;

% 计算每组各自的正确率
accuracy = zeros(nTest, 1);
for i=1:nTest
    beginID = 1 + (i-1)*nClass;
    endID   = beginID + nClass - 1;
    accuracy(i) = size(find(predict_label(beginID:endID)==test_label(beginID:endID)),1)/nClass;
end

% Result saving.
prefix = sprintf('result\\GCM_%dto%d_%s_%dsign_%dg',nDim, nSub,frameMode, nClass, nSample);
fileName_middleResult = [prefix num2str(testSample) '.mat'];
save(fileName_middleResult, 'VValKernel', 'accuracy', 'test_label', 'model_precomputed', '-v7.3');

fileName_result = [prefix '.txt'];
fid = fopen(fileName_result,'at+');
for i=1:nTest
    fprintf(fid,'P%02d\t%f\n',i+ID_testBegin-1, accuracy(i));
end
fprintf(fid, 'Cov generation time cost (s, 2000sign_12g): %f \n', covGeneTime_2000sign_12group);
fprintf(fid, 'Training time (s, 2000sign_8g): %f\n',trainTime);
fprintf(fid, 'Test time (s, 2000sign_4g): %f\n',testTime);
fclose(fid);