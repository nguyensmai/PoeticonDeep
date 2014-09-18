n_basis_functions = 10;
P = path();
P = path(P,'../../dmp_bbo_matlab_deprecated-master_deprecated/dynamicmovementprimitive');
path(P,'../../Autoencoder_Code');
digitdata=[];
targets=[];
nTargets = 9;


% display the numbers of the data
order = 3;
time =2;
dt=0.02;
time_exec=time;

for iDigit=1:9
  
    for iSample=1:nSamples(iDigit)
        coordsX1=data{iDigit,iSample}{1}(1);
        coordsY1=data{iDigit,iSample}{2}(1);
        coordsX2=data{iDigit,iSample}{1}(end);
        coordsY2=data{iDigit,iSample}{2}(end);
        
        trajectory.t = ( 0:1:( numel(data{iDigit,iSample}{1})-1) )';
        trajectory.y = [data{iDigit,iSample}{1}(:),data{iDigit,iSample}{2}(:)];
        yd = trajectory.y(2:end,:)-trajectory.y(1:end-1,:);
        d= size(yd,2);
        trajectory.yd= [zeros(1,d); yd];
        
        ydd = trajectory.yd(2:end,:)-trajectory.yd(1:end-1,:);
        trajectory.ydd = [zeros(1,d);ydd];
        
        
         figure(1)
         subplot(9,max(nSamples), (iDigit-1)*max(nSamples) +iSample)
         [ theta y0 g0 ] = dmptrain(trajectory,order,n_basis_functions);
         digitdata = [digitdata; coordsX1 coordsY1 coordsX2 coordsY2 theta(:)'];
        target = zeros(1,nTargets);
        target(iDigit) = 1;
         targets = [targets; target];
%         hold on
%         plot( data{iDigit,iSample}{1},data{iDigit,iSample}{2},'-r')
%         figure(2)
%         subplot(9,max(nSamples), (iDigit-1)*max(nSamples) +iSample)
%         [ trajectory ] = dmpintegrate([ coordsX1 coordsY1],[ coordsX2 coordsY2],theta,time,dt,time_exec,order);
%         hold on
%         handle = plot(trajectory.y(:,1,1),trajectory.y(:,2,1));
    end
end



% create batches
totnum=size(digitdata,1);
% randomorder=randperm(totnum);
numbatches=  2;
numdims  =  size(digitdata,2);
batchsize = totnum/numbatches;

batchdata = zeros(batchsize, numdims, numbatches);
batchtargets = zeros(batchsize, nTargets, numbatches);

maxdx = max(max(digitdata(:,5:4+n_basis_functions)));
mindx = min(min(digitdata(:,5:4+n_basis_functions)));
maxdy = max(max(digitdata(:,5+n_basis_functions:4+2*n_basis_functions)));
mindy = min(min(digitdata(:,5+n_basis_functions:4+2*n_basis_functions)));
batchdata1 = [
    digitdata(:,1:4)/500,...
    (digitdata(:,5:4+n_basis_functions)  -mindx)./(maxdx-mindx), ...
    (digitdata(:,5+n_basis_functions:4+2*n_basis_functions)-mindy)./(maxdy-mindy)
    ];
batchtargets1 = targets;
% for b=1:numbatches
%   batchdata(:,:,b) = digitdata(randomorder(1+(b-1)*batchsize:b*batchsize), :);
%   batchtargets(:,:,b) = targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
% end;
%%rr=randperm(totnum);
rr=[1:20:totnum 2:20:totnum 3:20:totnum 4:20:totnum 5:20:totnum...
    6:20:totnum 7:20:totnum 8:20:totnum 9:20:totnum 10:20:totnum...
    11:20:totnum 12:20:totnum 13:20:totnum 14:20:totnum 15:20:totnum...
    16:20:totnum 17:20:totnum 18:20:totnum 19:20:totnum 20:20:totnum];
batchdata1=batchdata1(rr,:);
batchtargets1=batchtargets1(rr,:);

for iBatch=1:numbatches
    batchdata(:,:,iBatch)=batchdata1((iBatch-1)*batchsize+1:iBatch*batchsize,:);
    batchtargets(:,:,iBatch)=batchtargets1((iBatch-1)*batchsize+1:iBatch*batchsize,:);
    
end
%%% Reset random seeds
rand('state',sum(100*clock));
randn('state',sum(100*clock));
save temp_batch
save maxmin maxdx mindx maxdy mindy n_basis_functions

% check the data batches
figure('name',num2str(n_basis_functions))
for iSample=1:batchsize
    coordsX1=batchdata(iSample,1,1);
    coordsY1=batchdata(iSample,2,1);
    coordsX2=batchdata(iSample,3,1);
    coordsY2=batchdata(iSample,4,1);
        
    visible= batchdata(iSample,5:end,1);
    visible = visible + 0.001*rand(size(visible)); %testing stability
    visible = visible.*[(maxdx-mindx)*ones(1,n_basis_functions) (maxdy-mindy)*ones(1,n_basis_functions)] ...
        +[mindx*ones(1,n_basis_functions) mindy*ones(1,n_basis_functions)];
    visible = reshape(visible,2,n_basis_functions);
    subplot(ceil(batchsize/nTargets), 9,iSample)
    iDigit=find(batchtargets(iSample,:,1)>0);
    [ trajectory ] = dmpintegrate([ coordsX1 coordsY1],...
        [coordsX2 coordsY2 ],visible,time,dt,time_exec,order);
    hold on
    handle = plot(trajectory.y(:,1,1),trajectory.y(:,2,1));
    title(find(batchtargets(iSample,:,1)>0))
end

%% create the dbn
maxepoch=1000;
numhid=1000; numpen=1000;
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




%% train each rbm with contrastive-divergence
% 
% maxepoch=1000;
% fprintf(1,'Pretraining Layer 1 with RBM: %d-%d \n',numdims,numhid);
% restart=1;
% [dbn.rbm{1},poshidstates, errL1] = rbmgaussian(batchdata,dbn.rbm{1},maxepoch,restart);
% title('layer1');
% save layer1
% 
% maxepoch=1000;
% fprintf(1,'\nPretraining Layer 2 with RBM: %d-%d \n',numhid,numpen);
% restart=1;
% [dbn.rbm{2}, batchpospenprobs,errL2] = rbmsigmoid(poshidstates,dbn.rbm{2},maxepoch, restart);
% title('layer2');
% save layer2;
% 
% maxepoch=1000;
% fprintf(1,'\nPretraining Layer 3  (hidden and labels) with RBM: [%d %d]-%d \n',numpen,nTargets,numpen2);
% restart=1;
% [dbn.rbm{3},errL3] = toprbm(batchpospenprobs,batchtargets,dbn.rbm{3},maxepoch,restart);
% save layer3;



%% up-down algorithm
dbn = untie(dbn);
[dbn, errLUD] = updown(batchdata,batchtargets,dbn, maxepoch);

%% testing
figure
for label=1:9
    visible = reshape(generativeModel(dbn,label).*(maxd-mind)+mind,2,n_basis_functions);
    subplot(3,3, label)
    [ trajectory ] = dmpintegrate([ 0 1],[ 1 0],visible,time,dt,time_exec,order);
    hold on
    handle = plot(trajectory.y(:,1,1),trajectory.y(:,2,1));
end