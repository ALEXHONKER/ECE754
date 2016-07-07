function [Pre,Rec,F1,ACCU]= read(name,id)
    path=strcat('/Users/jiangchenyang/Documents/workspace/ECE754/data/csvData/',name);
    path=strcat(path,num2str(id));
    trainpath=strcat(path,'_train.csv');
    testpath=strcat(path,'_test.csv');
    train0=csvread(trainpath,1,0);
    test0=csvread(testpath,1,0);
    FeaTrain=train0(:,1:(size(train0,2)-1));
    FeaTest=test0(:,1:(size(test0,2)-1));
    LabTrain=train0(:,size(train0,2));
    LabTest=test0(:,size(test0,2));
    [Pre,Rec,F1,ACCU] = DSUS(FeaTrain,FeaTest,LabTrain,LabTest)
end