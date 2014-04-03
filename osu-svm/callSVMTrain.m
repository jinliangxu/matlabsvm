function [AlphaY,SVs,Bias,Parameters,nSV,nLabel] = callSVMTrain(TrainData, TrainLabel, C, KernelType)
% Function  to call SVMTrain
% Usages:
% 		 [AlphaY,SVs,Bias,Parameters,nSV,nLabel] = callSVMTrain(TrainData, TrainLabel, C, KernelType)
%
% Description:
% 		Call SVM train function based on values of 
%		C (box constraint), Kernel type passed by
%		
% Inputs:
% 		TrainData		-	training data set, M*N (M row vector, M - number of features)
%		TrainLabel		-	training data set corresponding class type, 1*N (row vector)
%		C				-	values of C can be 1, 10 or 100
%		KernelType		-	which can be 0 - linear, 1 - polynomial, 2 - RBF
%
% Outputs:
% 		...... 			- 	refer to function SVMTrain.
%
% By Liang, Date 2014/4/2

%% Define SVM Kernel Hyperparameters
%C = 10;  % Define C (box constraint)
G = 2;  % Define Gamma = 1/Sigma (width of Rbf or order of polynomial)
%% "Grid search" ideally needed but not included here

%% Data verification 
% C should be 1, 10 or 100
if(C ~= 1 & C ~= 10 & C ~= 100)
	disp(' Value of C(box constraint) is not correct, can be only 1, 10 or 100.');
	return
end

%% Extract data(classes?)

%% Train SVM (Linear, Polynomial or RBF)
%% Train the Linear kernel SVM 
if(KernelType == 0)  
	[AlphaY,SVs,Bias,Parameters,nSV,nLabel] = LinearSVC(TrainData,TrainLabel,C);
%% Train the Polynomial kernel SVM 
elseif(KernelType == 1)
	[AlphaY,SVs,Bias,Parameters,nSV,nLabel] = PolySVC(TrainData,TrainLabel,G,C); 
%% Train the RBF kernel SVM 
elseif(KernelType == 2) 
	[AlphaY,SVs,Bias,Parameters,nSV,nLabel] = RbfSVC(TrainData,TrainLabel,G,C); 
%% Other cases, return error
else
	disp(' Error: Kernel type can be only: 0 - linear, 1 - polynomial, 2 - RBF.');
	return
end

%% Test output 
%[ClassRate, DecisionValue, Ns, ConfMatrix, PreLabels]= SVMTest(TestData, TestLabel, AlphaY, SVs, Bias, Parameters, nSV, nLabel);

%% Return results
%% NA

