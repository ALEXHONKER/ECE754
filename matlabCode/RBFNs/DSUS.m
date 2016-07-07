function [Pre,Rec, F1,ACCU] = DSUS(Tra,Tes,TraLab,TesLab)

%% Generate Points & Labels
% Features and Classes (Fr: feature train, Fs: feature test, Lr: label train, Ls: label test)  
%% STEP 1: Trainnig the initial RBFNN
UN=Tra(TraLab==0,:);  %negative item, majority 
UP=Tra(TraLab==1,:);   %positive items, minority
K               = floor(sqrt(size(UP,1))); % Number of Clusters (Number of Kernels)
P0=[];
R0=[];
KMI             = 50;                              % K-means Iteration

[ CENn, SIGMAn,DALn] = kmeansCluster( UN, K, KMI );
[ CENp, SIGMAp,DALp] = kmeansCluster( UP, K, KMI );
MINUSn=[];
MINUSp=[];
for i=1:K
    BOOLn=DALn(:,K+1)==i;
    TEMPn=DALn(BOOLn,:); %i th cluster, 
    FEATUREn=UN(BOOLn,:);
    [MINn,INDEXi]=min(TEMPn(:,i)); 
    R0=[R0;FEATUREn(INDEXi,:)];
    
    BOOLp=DALp(:,K+1)==i;
    TEMPp=DALp(BOOLp,:);
    FEATUREp=UP(BOOLp,:);
    [MINp,INDEXi]=min(TEMPp(:,i));
    P0=[P0;FEATUREp(INDEXi,:)];
end
tn=ismember(UN,R0);
tn=tn(:,1);
tp=ismember(UP,P0);
tp=tp(:,1);
tempUN=UN(tn,:);
tempUP=UP(tp,:);
S=[tempUN;tempUP];
UN=UN(~tn,:);
UP=UP(~tp,:);
b=0;
LabelS=[zeros(size(tempUN,1),1);ones(size(tempUP,1),1)];
%% train RBFNN
[W, MU, SIGMA]  = rbfn_train(S, LabelS, K, KMI);      % train RBFNs
K
while size(UP,1)>=K
    C=[];
    RB=[];
    PB=[];
    np_size=size(UP,1)
    b=b+1;
    [ CEN, SIGMATEMP,DAL] = kmeansCluster( UN, np_size, KMI );
    for i=1:np_size
        BOOL=DAL(:,np_size+1)==i;
        TEMP=DAL(BOOL,:); %i th cluster, 
        FEATURE=UN(BOOL,:);
        [MIN,INDEX]=min(TEMP(:,i)); 
        C=[C;FEATURE(INDEX,:)];
    end
    %% Sample Selection using the SM
    Q= mean(mean([C;UP]));%%how to choose Q??
    SMc=computeSM(C,W,K,MU',SIGMA,Q);
    SMUp=computeSM(UP,W,K,MU',SIGMA,Q);
    SMc=[SMc,[1:size(SMc,1)]'];
    SMUp=[SMUp,[1:size(SMUp,1)]'];
    SMc=sortrows(SMc);
    SMUp=sortrows(SMUp);
    RB=C(SMc(size(SMc,1)-K+1:size(SMc,1),2),:);
    PB=UP(SMUp(size(SMUp,1)-K+1:size(SMUp,1),2),:);
    tn=ismember(UN,RB);
    tn=tn(:,1);
    tp=ismember(UP,PB);  %% BUG!!!!!!!!!!!!
    tp=tp(:,1);
    tempUN=UN(tn,:);
    tempUP=UP(tp,:);
    S=[S;tempUN;tempUP];
    UN=UN(~tn,:);
    UP=UP(~tp,:);
    LabelS=[LabelS;zeros(size(tempUN,1),1);ones(size(tempUP,1),1)];
    [W, MU, SIGMA]  = rbfn_train(S, LabelS, K, KMI);
end

Y = rbfn_test(Tes, W, K, MU, SIGMA);  % test RBFNs
TP= sum((TesLab(:,1)==1) & (TesLab(:,1)==Y(:,1)))
TN= sum(TesLab(:,1)==0 & TesLab(:,1)==Y(:,1))
FP =  sum(TesLab(:,1)==0 & TesLab(:,1)~=Y(:,1))
FN =  sum(TesLab(:,1)==1 & TesLab(:,1)~=Y(:,1))
Pre = TP/(TP+FP);
Rec = TP/(TP+FN);
F1= 2*Pre*Rec/(Pre+Rec);
ACCU   = 1 - sum(abs(Y-TesLab))/size(Y,1);
end

