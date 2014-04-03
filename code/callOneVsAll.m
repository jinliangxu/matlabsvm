function [TrainAcc, TestAcc] = callOneVsAll(DataSet, LabelSet, FoldNum)
% Function to split data set and calculate classification accuracy of one against all SVM.
% Usages:
% 		 [TrainAcc, TestAcc] = callOneVsAll(DataSet, LabelSet, FoldNum)
%
% Description:
% 		Split data set randomly into N-fold parts
%		Perform N-fold one against all SVM
%		
% Inputs:
% 		DataSet			-	all data set, M*N (M row vector, M - number of features)
%		LabelSet		-	all data set corresponding class type, 1*N (row vector)
% 		FoldNum			-	fold number
%
% Outputs:
% 		* TrainAccuracy	- 	training accuracy
%		* TestAccuracy 	- 	testing accuracy
%
% By Liang, Date 2014/4/2
 

%% Input data validation check 
if (nargin ~= 3)
   disp(' Error: Incorrect number of input variables.');
   help callOneVsAll;
   return
end
%% Data set 
[dataM, dataN]	 = size(DataSet);
[labelM, labelN] = size(LabelSet);
if (dataN ~= labelN | labelM ~= 1)
	disp(' Error: DataSet or LabelSet not correct.');
	disp('        DataSet - M*N, LabelSet - 1*N.');
	return	
end

%% Data processing
%% Preprocessing
[dumB,dumI,proLabels]	= unique(LabelSet);
%proData 				= zscore(DataSet);  %M*N              
proData 				= DataSet;
%data = DataSet;
numSet		= dataN;           % e.g. 150 here
numLabels	= max(proLabels);         % number of labels used

%% Random splitting 
randIdxes	= randperm(numSet);
%% dataCell{cellId,1} and labelCell{cellId,1}
dataCell 	= cell(FoldNum, 1);
labelCell 	= cell(FoldNum, 1); 
cellSize	= fix(numSet/FoldNum);

%% iterate and append M*1 data to dataCell{cellId,1} each round
%% finally dataCell should have FoldNum of cells
%% each dataCell{cellId,1} should be a matrix of M*cellSize
for i = 1:numSet  
	cellId = fix((i-1)/cellSize)+1;  %% 1..150 [1..30 31..60 61..90 91..120 121..150]
	%% Add all left data to the last data/label cell in case numSet cannot be devided by FoldNum
	if (cellId > FoldNum) 
		dataCell{FoldNum, 1} 	= [dataCell{FoldNum, 1} proData(:,randIdxes(i))];
		labelCell{FoldNum, 1} 	= [labelCell{FoldNum, 1} proLabels(randIdxes(i))];
	else
		dataCell{cellId, 1} 	= [dataCell{cellId, 1} proData(:,randIdxes(i))];
		labelCell{cellId, 1} 	= [labelCell{cellId, 1} proLabels(randIdxes(i))];
	end
end
%% now we have 1..FoldNum of dataset and label set, which D1..Dx, L1..Lx

%% generate train data and test data for each fold/round
%% perform one against all svm and return result  of accuracy
accTrainCell = cell(FoldNum,1);
accTestCell = cell(FoldNum,1);	
for j = 1:FoldNum
	%% prepare train data and test data
	trainData 	= [];
	trainLabel 	= [];
	testData	= dataCell{j,1};
	testLabel	= labelCell{j,1}; 
	%% numTrain, numTest
	for k = 1:FoldNum
		if (j ~= k)
			trainData  = [trainData dataCell{k,1}];	  %%M*x
			trainLabel = [trainLabel labelCell{k,1}]; %%1*X
		end
	end
	
	%% now we have generated train data and test data 
	%% next perform one against all svm 
	%% [TrainAccuracy, TestAccuracy] = callOneVsAllSVM(TrainData, TestData, TrainLabel, TestLabel, NumLabels, C, KernelType);
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%AC  % Linear % Poly % RBF %%
	%C1  %   -    %  -   %  -  %%
	%C10 %   -    %  -   %  -  %%
	%C100%   -    %  -   %  -  %%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	accTrainMatrix = zeros(3,3);
	accTestMatrix  = zeros(3,3);
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	C = 1;
	for ( cId = 1:3)         %% 1, 10, 100
		for kernelType = 0:2 %% 0 linear, 1 poly, 2 Rbf
			[TrainAccuracy, TestAccuracy] = callOneVsAllSVM(trainData, testData, trainLabel, testLabel, numLabels, C, kernelType);
			accTrainMatrix(cId, kernelType+1) = TrainAccuracy;  
			accTestMatrix(cId, kernelType+1)  = TestAccuracy;
		end
		C = C*10;            %% 1, 10, 100
	end
	accTrainCell{j,1} = accTrainMatrix;
	accTestCell{j,1}  = accTestMatrix;
end
 
%% return RES
TrainAcc = accTrainCell;
TestAcc  = accTestCell;

%% NA