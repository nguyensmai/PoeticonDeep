P = path();
P = path(P,'../../dmp_bbo_matlab_deprecated-master_deprecated/dynamicmovementprimitive');
path(P,'../../Autoencoder_Code');
digitdata=[];
targets=[];
nTargets = 9;


%% display the numbers of the data
order = 3;
n_basis_functions = 100;

for iDigit=1:9
    for iSample=1:nSamples(iDigit)
        
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
        digitdata = [digitdata; theta(:)'];
        target = zeros(1,nTargets);
        target(iDigit) = 1;
        targets = [targets; target];
        hold on
        plot( data{iDigit,iSample}{1},data{iDigit,iSample}{2},'-r')
        figure(2)
        subplot(9,max(nSamples), (iDigit-1)*max(nSamples) +iSample)
        [ trajectory ] = dmpintegrate([ 0 1],[ 1 0],theta,time,dt,time_exec,order);
        hold on
        handle = plot(trajectory.y(:,1,1),trajectory.y(:,2,1));
    end
end


%% create batches
totnum=size(digitdata,1);
randomorder=randperm(totnum);
numbatches=  1;
numdims  =  size(digitdata,2);
batchsize = totnum;

batchdata = zeros(batchsize, numdims, numbatches);
batchtargets = zeros(batchsize, 10, numbatches);

maxd = max(digitdata);
mind = min(digitdata);
batchdata = (digitdata-repmat(mind,batchsize,1))./(repmat(maxd,batchsize,1)-repmat(mind,batchsize,1));
batchtargets = targets;
% for b=1:numbatches
%   batchdata(:,:,b) = digitdata(randomorder(1+(b-1)*batchsize:b*batchsize), :);
%   batchtargets(:,:,b) = targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
% end;

%%% Reset random seeds
rand('state',sum(100*clock));
randn('state',sum(100*clock));
save temp_batch


%% create the dbn
maxepoch=2000;
numhid=500; numpen=1000; numpen2=1000;
nodes = [numdims numhid numpen numpen2];
dbn = randDBN(nodes,nTargets,'GBRBM');


%% train each layer with rbm

maxepoch=6000;
fprintf(1,'Pretraining Layer 1 with RBM: %d-%d \n',numdims,numhid);
restart=1;
[dbn.rbm{1},batchposhidprobs] = rbmgaussian(batchdata,dbn.rbm{1},maxepoch,restart);
title('layer1');
save layer1 dbn

maxepoch=3000;
fprintf(1,'\nPretraining Layer 2 with RBM: %d-%d \n',numhid,numpen);
restart=1;
[dbn.rbm{2}, batchpospenprobs] = rbmsigmoid(batchposhidprobs,dbn.rbm{2},maxepoch, restart);
title('layer2');
save layer2 dbn;

maxepoch=200;
fprintf(1,'\nPretraining Layer 3  (hidden and labels) with RBM: [%d %d]-%d \n',numpen,nTargets,numpen2);
restart=1;
[dbn.rbm{3}] = toprbm(batchpospenprobs,batchtargets,dbn.rbm{3},maxepoch,restart);
save layer3 dbn;



%%
figure
for epoch=1:100
    [dbn, errsum, err1,err2] = updown(batchdata,targets,dbn);
    plot(epoch, errsum,'xb');
    hold on; 
    plot(epoch, err1,'xg');
    plot(epoch, err2,'xr');
end


%% testing
figure(3)
for label=1:9
visible = generativeModel(dbn,label)
subplot(3,3, label)
[ trajectory ] = dmpintegrate([ 0 1],[ 1 0],visible,time,dt,time_exec,order);
hold on
handle = plot(trajectory.y(:,1,1),trajectory.y(:,2,1));
end