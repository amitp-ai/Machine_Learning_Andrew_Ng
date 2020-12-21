%Program to build my own dataset (optional exercise)
%The program will create an array of all the files in the directory
%Then will open each file individually

clear, close all, clc;
addpath C:\Users\amit_p\Desktop\SVM-ex6\ex6\Spam_Emails

fileList = dir('C:\Users\amit_p\Desktop\SVM-ex6\ex6\Spam_Emails');
fileList = fileList(~[fileList.isdir]); %remove any directories
[junk, sortorder] = sort([fileList.datenum]); %defaults to ascending order
fileList = fileList(sortorder); 

numFiles = 3; %numel(fileList); %number of files in the directory
filesCell = cell(numFiles, 1); %a cell for the contents of all the files
X = []; %initialize input features to an empty array
word_indices = {};
actualWords = {};

for ii=1:numFiles
    filesCell{ii} = readFile(fileList(ii).name);
    [actualWords{ii}, word_indices{ii}]  = processEmail_AmitSpamDataset(filesCell{ii});
    
    %[str{ii}, word_indices{ii}]  = processEmail(filesCell{ii});
    %X = [X; (emailFeatures(word_indices{ii}))']; %to make sure examples are in the rows
end
AmitVocabList = amitGenerateVocab(actualWords);

%%%code to randomly shuffle X
m = size(X, 1);
tempInd = 0;
tempVal = 0;
for ii = 1:m
    tempInd = ceil(rand()*m);
    tempVal = X(tempInd,:);
    X(tempInd,:) = X(ii,:);
    X(ii,:) = tempVal;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y = ones(size(X,1), 1); %create a y vector. It'll be all 1 since the data is spam.

save('AmitDatasetSpam.mat', 'X', 'y'); %saves the features variable in to a .mat file
