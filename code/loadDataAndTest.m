function [] = loadDataAndTest()
% Function to load data set and perform 5 fold one against all SVM
% Usage:
% 		 [] = loadDataAndTest()
%
% Description:
% 		Load test data set and perform 5 fold one against all SVM
%		Print out all accuracy result
%		
% Inputs:
% 		Make sure you have data set stored in the correct place
% 		and put all matlab script files in OSU-SVM lib directory.
%
% Outputs:
% 		* TrainAccuracy	- 	training accuracy
%		* TestAccuracy 	- 	testing accuracy
%
% By Liang, Date 2014/4/2 

%% Clean ENV 
clear all
clc
%% load data set
%load fisheriris
load('E:\proj\matlabsvm\data\iris_class1_2_3_4D.mat');
% fisherIris
% X 4*150
% t 150*1

%% prepare data set
dataSet 	= X;
labelSet 	= t';
foldNum 	= 5;
%trainAcc 	= cell(foldNum,1);
%testAcc 	= cell(foldNum,1);
[trainAcc, testAcc] = callOneVsAll(dataSet, labelSet, foldNum);
%trainAcc
%trainAcc{1,1}
%trainAcc{1,1}(1,1)
%% each cell
%% Fold - cell 1..foldNum 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%AC  % Linear % Poly % RBF %%
%C1  %   -    %  -   %  -  %%
%C10 %   -    %  -   %  -  %%
%C100%   -    %  -   %  -  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C = 1;
for ( cId = 1:3)         %% 1, 10, 100
	trainSum = zeros(1,3); %% linear, poly, Rbf
	testSum  = zeros(1,3);
	disp(sprintf('\n One against all SVM accuracy, C = %i.', C));
	disp(' Fold Linear(Train, Test)  Poly(Train, Test)   RBF(Train, Test)');
	for (fold = 1:foldNum) 
		for (kId = 1:3)  %% 0 linear, 1 poly, 2 Rbf
			trainSum(kId) = trainSum(kId) + trainAcc{fold,1}(cId,kId);
			testSum(kId) = testSum(kId) + testAcc{fold,1}(cId,kId);
		end
		%trainLinearAcc = trainAcc{fold,1}(cId, kId);
		disp(sprintf(' (%i)  |%f|%f| |%f|%f| |%f|%f|', fold, trainAcc{fold,1}(cId, 1),testAcc{fold,1}(cId, 1), trainAcc{fold,1}(cId, 2),testAcc{fold,1}(cId, 2), trainAcc{fold,1}(cId, 3),testAcc{fold,1}(cId, 3)));
	end
	
	disp(sprintf(' Avg  |%f|%f| |%f|%f| |%f|%f|',trainSum(1)/foldNum,testSum(1)/foldNum, trainSum(2)/foldNum,testSum(2)/foldNum, trainSum(3)/foldNum,testSum(3)/foldNum));
	C = C*10;
end
