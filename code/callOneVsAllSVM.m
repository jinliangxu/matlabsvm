function [TrainAccuracy, TestAccuracy] = callOneVsAllSVM(TrainData, TestData, TrainLabel, TestLabel, NumLabels, C, KernelType)
% Function to do one against all SVM.
% Usage:
% 		 [TrainAccuracy, TestAccuracy] = callOneVsAllSVM(TrainData, TestData, TrainLabel, TestLabel, NumLabels, C, KernelType)
%
% Description:
% 		Use one against all way to  do multi-class classification
%		
% Inputs:
% 		TrainData		-	training data set, M*N (M row vector, M - number of features)
%		TrainLabel		-	training data set corresponding class type, 1*N (row vector)
% 		TestData		-	testing data set, M*n
%		TestLabel		-	testing data set corresponding class type, 1*n
%		NumLabels 		- 	number of labels for testing data set
%		C				-	values of C can be 1, 10 or 100
%		KernelType		-	which can be 0 - linear, 1 - polynomial, 2 - RBF
%
% Outputs:
% 		TrainAccuracy	- 	training accuracy
%		TestAccuracy 	- 	testing accuracy
%
% By Liang, Date 2014/4/2
 
 
%% one against all 

%% train one-against-all models
numTrain 	= size(TrainLabel, 2);
numTest 	= size(TestLabel, 2);
trainProb 	= zeros(NumLabels, numTrain);
testProb 	= zeros(NumLabels, numTest);
for m=1:NumLabels
	[AlphaY,SVs,Bias,Parameters,nSV,nLabel] = callSVMTrain(TrainData, double(TrainLabel==m), C, KernelType);
	%%[ClassRate, DecisionValue, Ns, ConfMatrix, PreLabels] = SVMTest(...); %%PreLabels 1*N
	%% Train Data Prediction
	[trainCR, trainDV, trainNS, trainCM, trainPL] = SVMTest(TrainData, double(TrainLabel==m), AlphaY, SVs, Bias, Parameters, nSV, nLabel);
	%% Test Data Prediction
	[testCR, testDV, testNS, testCM, testPL] = SVMTest(TestData, double(TestLabel==m), AlphaY, SVs, Bias, Parameters, nSV, nLabel);
	%% Calc probability of class==m
	%trainProb(m,:) = abs(trainDV.*double(TrainLabel==m)); 
	%testProb(m,:) = abs(testDV.*double(TestLabel==m)); 
	trainProb(m,:) = abs(trainDV.*trainPL); 
	testProb(m,:) = abs(testDV.*testPL); 
end

%% predict the class with the highest probability
[dummyTrain,trainPred] = max(trainProb,[],1);
[dummyTest,testPred] = max(testProb,[],1);

%% calculate classification accuracy
TrainAccuracy = sum(trainPred == TrainLabel) ./ numel(TrainLabel);
TestAccuracy = sum(testPred == TestLabel) ./ numel(TestLabel);

%% return value of accuracy
%% NA