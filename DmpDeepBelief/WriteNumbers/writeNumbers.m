
n_basis_functions = 50;
P = path();
P = path(P,'../../dmp_bbo_matlab_deprecated-master_deprecated/dynamicmovementprimitive');
path(P,'../../Autoencoder_Code');
digitdata=[];
targets=[];
nTargets = 9;


%% display the numbers of the data
order = 3;

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
% randomorder=randperm(totnum);
numbatches=  1;
numdims  =  size(digitdata,2);
batchsize = totnum;

batchdata = zeros(batchsize, numdims, numbatches);
batchtargets = zeros(batchsize, nTargets, numbatches);

maxdx = max(max(digitdata(:,1:50)));
mindx = min(min(digitdata(:,1:50)));
maxdy = max(max(digitdata(:,51:100)));
mindy = min(min(digitdata(:,51:100)));
batchdata = [
    (digitdata(:,1:50)  -mindx)./(maxdx-mindx), ...
    (digitdata(:,51:100)-mindy)./(maxdy-mindy)
    ];
batchtargets = targets;
% for b=1:numbatches
%   batchdata(:,:,b) = digitdata(randomorder(1+(b-1)*batchsize:b*batchsize), :);
%   batchtargets(:,:,b) = targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
% end;

%%% Reset random seeds
rand('state',sum(100*clock));
randn('state',sum(100*clock));
save temp_batch

%% check the data batches
figure('name',num2str(n_basis_functions))
for iSample=1:totnum
    visible= batchdata(iSample,:);
    visible = visible + 0.01*rand(size(visible)); %testing stability
    visible = visible.*[(maxdx-mindx)*ones(1,n_basis_functions) (maxdy-mindy)*ones(1,n_basis_functions)] ...
        +[mindx*ones(1,n_basis_functions) mindy*ones(1,n_basis_functions)];
    visible = reshape(visible,2,n_basis_functions);
    subplot(9,ceil(totnum/9), iSample)
    iDigit=find(batchtargets(iSample,:)>0);
    [ trajectory ] = dmpintegrate([ data{iDigit,1}{1}(1) data{iDigit,1}{2}(1)],...
        [data{iDigit,1}{1}(end) data{iDigit,1}{1}(end) ],visible,time,dt,time_exec,order);
    hold on
    handle = plot(trajectory.y(:,1,1),trajectory.y(:,2,1));
    title(find(batchtargets(iSample,:)>0))
end

%% create the dbn
maxepoch=1000;
numhid=50; numpen=50; numpen2=100;
nodes = [numdims numhid numpen numpen2];
dbn = randDBN(nodes,nTargets,'GBRBM');


%% train each rbm with contrastive-divergence

maxepoch=1000;
fprintf(1,'Pretraining Layer 1 with RBM: %d-%d \n',numdims,numhid);
restart=1;
[dbn.rbm{1},poshidstates, errL1] = rbmgaussian(batchdata,dbn.rbm{1},maxepoch,restart);
title('layer1');
save layer1

maxepoch=1000;
fprintf(1,'\nPretraining Layer 2 with RBM: %d-%d \n',numhid,numpen);
restart=1;
[dbn.rbm{2}, batchpospenprobs,errL2] = rbmsigmoid(poshidstates,dbn.rbm{2},maxepoch, restart);
title('layer2');
save layer2;

maxepoch=1000;
fprintf(1,'\nPretraining Layer 3  (hidden and labels) with RBM: [%d %d]-%d \n',numpen,nTargets,numpen2);
restart=1;
[dbn.rbm{3},errL3] = toprbm(batchpospenprobs,batchtargets,dbn.rbm{3},maxepoch,restart);
save layer3;



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