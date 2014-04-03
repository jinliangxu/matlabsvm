
%# load data
%# split data to 5 sets
%# cross-validation




%# Fisher Iris dataset
load fisheriris
[dummy,dummy,labels] = unique(species);   %# labels: 1/2/3
data = zscore(meas);              %# scale features
numInst = size(data,1);           % e.g. 150 here
numLabels = max(labels);          % 1,2,3 here is 3, types used

%# split training/testing
%# should divide to 5 parts
idx = randperm(numInst);
numTrain = 100; 
numTest = numInst - numTrain;  %150 -50 =50

transdata = data';
translabels = labels';
trainData = transdata(:,idx(1:numTrain)); %% 4x100
trainLabel = translabels(idx(1:numTrain)); %%

testData = transdata(:,idx(numTrain+1:end)); %%4x50
testLabel = translabels(idx(numTrain+1:end));

%# train one-against-all models
model = cell(numLabels,1); %cell(3,1)
prob = zeros(numLabels,numTest);
for k=1:numLabels
	%    Samples    - training samples, MxN, (a row of column vectors);
	%    Labels     - labels of training samples, 1xN, (a row vector);
	% [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = SVMTrain(Samples, Labels, Parameters)
	%    model{k}
	%[AlphaY, SVs, Bias, Parameters, nSV, nLabel]	= svmtrain(trainData, double(trainLabel==k), '-c 1 -g 0.2 -b 1');
	[AlphaY,SVs,Bias,Parameters,nSV,nLabel] = callSVMTrain(trainData, double(trainLabel==k), 1, 0);
    [ClassRate, dv, Ns, ConfMatrix, p] = SVMTest(testData, double(testLabel==k), AlphaY, SVs, Bias, Parameters, nSV, nLabel);
    %prob(k,:) = dv(:,p==1);	%# probability of class==k
    prob(k,:) = abs(dv.*p);	%# probability of class==k
	%k
	%dv % decision value
	%ClassRate 
	%Ns
	%p %predicted labels
	%double(testLabel==k)
	%double(testLabel==k)
	%sum(double(testLabel==k))/numel(testLabel)
	%% prob(:,k) = p(:,model{k}.Label==1);    %# probability of class==k
	%% num*3 - prob, model{k}.Label - num*1
	%double(p==1)
	%double(testLabel==k)
	% - dv的值是如何偏差的 
	% - 根据dv组成dv矩阵 取最大、最小的进行
end
%prob

%# get probability estimates of test instances using each model
%prob = zeros(numTest,numLabels);
%for k=1:numLabels
	%[predicted_label, accuracy, decision_values]= svmpredict(test_class,test_data, model);
	%[ClassRate, DecisionValue, Ns, ConfMatrix, PreLabels]= SVMTest(test_data, test_class, AlphaY, SVs, Bias, Parameters, nSV, nLabel);
	
%    [ClassRate, DecisionValue, Ns, ConfMatrix, p] = SVMTest(testData,double(testLabel==k), model{k}, '-b 1');
%    prob(:,k) = p(:,model{k}.Label==1);    %# probability of class==k
%end

%# predict the class with the highest probability
[dummy,pred] = max(prob,[],1);  %numTest*1 
acc = sum(pred == testLabel) ./ numel(testLabel)    %# accuracy
%C = confusionmat(testLabel, pred)                   %# confusion matrix
