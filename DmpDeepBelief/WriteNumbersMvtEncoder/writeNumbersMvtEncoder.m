% Copyright (c) 2013 Sao Mai Nguyen
%               e-mail : nguyensmai@gmail.com
%               http://nguyensmai.free.fr/%%
P = path();
P = path(P,'../../dmp_bbo_matlab_deprecated-master_deprecated/dynamicmovementprimitive');
path(P,'../../Autoencoder_Code');
digitdata=[];
targets=[];
nTargets = 10;

%% replicate data
for iDigit=1:nTargets
        for iSample=1:nSamples(iDigit)
            delta=randi(15,1,2)-8;
            data{iDigit,nSamples(iDigit)+iSample}{1}=data{iDigit,iSample}{1}+delta(1);
            data{iDigit,nSamples(iDigit)+iSample}{2}=data{iDigit,iSample}{2}+delta(2);
        end
end

nSamples=2*nSamples;

%% check nan values
for iDigit=1:nTargets
    for iSample=1:nSamples(iDigit)
        if any(isnan(data{iDigit,iSample}{1})) || any(isnan(data{iDigit,iSample}{2}))
            disp(['error ',num2str(iDigit),' ',num2str(iSample)]);
        end
    end
end

%% convert data to the appropriate format
tmax=-1;
counter=1;

%%
for iDigit=4:nTargets
    parfor iSample=3251:nSamples(iDigit)
        tmax=length(data{iDigit,iSample}{1});
        y1=data{iDigit,iSample}{1}';
        y2=data{iDigit,iSample}{2}';
        t=[0:tmax-1]';
        try
        [a1, resnorm1] = MovementEncoder.getMovementParameters(t,y1);
        [a2, resnorm2] = MovementEncoder.getMovementParameters(t,y2);
        catch
            try
                y1=y1+randi(11)-6;
                y2=y2+randi(11)-6;
                [a1, resnorm1] = MovementEncoder.getMovementParameters(t,y1);
                [a2, resnorm2] = MovementEncoder.getMovementParameters(t,y2);
            catch
                y1=y1+randi(11)-6;
                y2=y2+randi(11)-6;
                [a1, resnorm1] = MovementEncoder.getMovementParameters(t,y1);
                [a2, resnorm2] = MovementEncoder.getMovementParameters(t,y2);
                
            end
            
        end
        Vinv1 = MovementEncoder.getMovementTrajectory(a1,t);
        Vinv2 = MovementEncoder.getMovementTrajectory(a2,t);
%         digitdata(sum(nSamples(1:iDigit-1))+iSample,1:118) = [ a1 a2];
digitdataC{iDigit,iSample} = [ a1 a2];
        target = zeros(1,nTargets);
        target(iDigit) = 1;
        targets = [targets; target];
         
       % subplot(nTargets,3, counter)
%         subplot(nTargets,max(nSamples), (iDigit-1)*max(nSamples) +iSample)
%         plot(y1,y2,'-');
%         hold on
%          plot(a1,a2,'or');
%          hold on
%         plot(Vinv1,Vinv2, 'Xb')
        counter=counter+1;
    end
end

% create batches
totnum=size(digitdata,1);
% randomorder=randperm(totnum);
numbatches=  5;
numdims  =  size(digitdata,2);
batchsize = totnum/numbatches;

batchdata = zeros(batchsize, numdims, numbatches);
batchtargets = zeros(batchsize, nTargets, numbatches);

maxdx = max(max(digitdata(:,1:numdims/2)));
mindx = min(min(digitdata(:,1:numdims/2)));
maxdy = max(max(digitdata(:,numdims/2+1:end)));
mindy = min(min(digitdata(:,numdims/2+1:end)));
batchdata1 = [
    (digitdata(:,1:numdims/2)  -mindx)./(maxdx-mindx), ...
    (digitdata(:,numdims/2+1:end)-mindy)./(maxdy-mindy)
    ];
batchtargets1 = targets;
% for b=1:numbatches
%   batchdata(:,:,b) = digitdata(randomorder(1+(b-1)*batchsize:b*batchsize), :);
%   batchtargets(:,:,b) = targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
% end;
%%rr=randperm(totnum);
rr=[1:21:totnum 2:21:totnum 3:21:totnum 4:21:totnum 5:21:totnum...
    6:21:totnum 7:21:totnum 8:21:totnum 9:21:totnum 10:21:totnum...
    11:21:totnum 12:21:totnum 13:21:totnum 14:21:totnum 15:21:totnum...
    16:21:totnum 17:21:totnum 18:21:totnum 19:21:totnum 20:21:totnum 21:21:totnum];
batchdata1=batchdata1(rr,:);
batchtargets1=batchtargets1(rr,:);

for iBatch=1:numbatches
    batchdata(:,:,iBatch)=batchdata1((iBatch-1)*batchsize+1:iBatch*batchsize,:);
    batchtargets(:,:,iBatch)=batchtargets1((iBatch-1)*batchsize+1:iBatch*batchsize,:);
    
end
%%% Reset random seeds
rand('state',sum(100*clock));
randn('state',sum(100*clock));
save temp_batchMvtEncoder
save maxmin maxdx mindx maxdy mindy n_basis_functions

%% check the data batches
figure('name',num2str(n_basis_functions))
t=[0:500]';
for iSample=1:batchsize
        
    visible= batchdata(iSample,:,1);
    visible = visible + 0.000*rand(size(visible)); %testing stability
    visible = visible.*[(maxdx-mindx)*ones(1,numdims/2) (maxdy-mindy)*ones(1,numdims/2)] ...
        +[mindx*ones(1,numdims/2) mindy*ones(1,numdims/2)];
    a1 = visible(:,1:numdims/2);
    a2 = visible(:,numdims/2+1:end);
    subplot(ceil(batchsize/nTargets), nTargets,iSample)
    iDigit=find(batchtargets(iSample,:,1)>0);
     Vinv1 = MovementEncoder.getMovementTrajectory(a1,t);
     Vinv2 = MovementEncoder.getMovementTrajectory(a2,t);
     handle= plot(Vinv1,Vinv2, 'Xb');
    hold on
    title(find(batchtargets(iSample,:,1)>0))
end



%% create the dbn
maxepoch=1000;
numhid=200; numpen=1000;
nodes = [numdims numhid numpen];
dbn = randDBN(nodes,nTargets,'ASSOC');

%% train DBM
maxepoch=500001;
fprintf(1,'Pretraining Layer 1 with RBM: %d-%d \n',numdims,numhid);
restart=1;
[dbn.rbm{1},batchposhidprobs, errL1,negdata] = rbmsigmoid(batchdata,dbn.rbm{1},maxepoch,restart);
title('error for layer1');
save layer1
show_rbm(negdata(1:81,:,1),numdims)
title('reconstruction by layer1')


maxepoch=500001;
%fprintf(1,'\nPretraining Layer 3  (hidden and labels) with RBM: [%d %d]-%d \n',numpen,nTargets,numpen2);
restart=1;
[dbn,errL3] = toprbm(batchdata,batchtargets,dbn,maxepoch,restart);
title('error for layer3');
save layer3;



%% mean-field
dbn = untie(dbn);
maxepoch=6001;
restart=0;
[dbn, errBack, negstates, neglabels] = dbm_mf(batchdata,batchtargets,dbn, maxepoch,restart);
title('error for mean-field');
save dbmMfTrain

figure
[dbn,  negstates] = generativeModel(batchdata,batchtargets,eye(nTargets),dbn,500)
figure
[~, errTest, negstates, neglabels] = dbm_mf(batchdata,batchtargets,dbn, 1,0);
