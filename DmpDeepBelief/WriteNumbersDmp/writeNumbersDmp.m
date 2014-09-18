% Copyright (c) 2013 Sao Mai Nguyen
%               e-mail : nguyensmai@gmail.com
%               http://nguyensmai.free.fr/%%

%%

n_basis_functions = 10;
P = path();
P = path(P,'../../dmp_bbo_matlab_deprecated-master_deprecated/dynamicmovementprimitive');
path(P,'../../Autoencoder_Code');
digitdata=[];
targets=[];
nTargets = 10;
nRepeat=1;

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


%% display the numbers of the data
order = 3;
time =2;
dt=0.02;
time_exec=time;

for iDigit=1:nTargets
    for iSample=1:nSamples(iDigit)
        try
            data{iDigit,iSample}{1}(1);
        catch
            data{iDigit,iSample}{1}=(0.9+0.2*rand(1))*data{iDigit,iSample-1}{1}+randi(15)-8;
            data{iDigit,iSample}{2}=(0.9+0.2*rand(1))*data{iDigit,iSample-1}{2}+randi(15)-8;
        end
        
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
        
        
        fig2= figure(2);
        %           subplot(nTargets,max(nSamples), (iDigit-1)*max(nSamples) +iSample)
        hold on
        [ theta y0 g0 ] = dmptrain(trajectory,order,n_basis_functions);
        if any(isnan(theta(:)))
            data{iDigit,iSample}{1}=(0.9+0.2*rand(1))*data{iDigit,iSample-1}{1}+randi(15)-8;
            data{iDigit,iSample}{2}=(0.9+0.2*rand(1))*data{iDigit,iSample-1}{2}+randi(15)-8;
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
            
            
            fig2= figure(2);
            %           subplot(nTargets,max(nSamples), (iDigit-1)*max(nSamples) +iSample)
            hold on
            [ theta y0 g0 ] = dmptrain(trajectory,order,n_basis_functions);
        end
        %          theta=max(min(theta,10^10),-10^10);
        if any(theta(:)>10^10)||any(theta(:)<-10^10)
            disp(['too big value :', num2str(iDigit),' ' , num2str(iSample)])
        end
        digitdata = [digitdata; [coordsX1 coordsY1 coordsX2 coordsY2 theta(:)']];
        target = zeros(1,nTargets);
        target(iDigit) = 1;
        targets = [targets; target];
        hold on
        plot( data{iDigit,iSample}{1},data{iDigit,iSample}{2},'-r')
        %figure(2)
        %  subplot(6,6, iSample)
        %         [ trajectory ] = dmpintegrate([ coordsX1 coordsY1],[ coordsX2 coordsY2],theta,time,dt,time_exec,order);
        %         hold on
        %         handle = plot(trajectory.y(:,1,1),trajectory.y(:,2,1));
    end
end



% create batches
totnum=size(digitdata,1);
% randomorder=randperm(totnum);
numbatches=  1;
numdims  =  size(digitdata,2);
batchsize = totnum/numbatches;

batchdata = zeros(batchsize, nRepeat*numdims, numbatches);
batchtargets = zeros(batchsize, nTargets, numbatches);

maxdx = (max(digitdata(:,5:4+n_basis_functions)));
mindx = (min(digitdata(:,5:4+n_basis_functions)));
maxdy = (max(digitdata(:,5+n_basis_functions:4+2*n_basis_functions)));
mindy = (min(digitdata(:,5+n_basis_functions:4+2*n_basis_functions)));
batchdata1 = [
    digitdata(:,1:4)/500,...
    (digitdata(:,5:4+n_basis_functions)  -repmat(mindx,totnum,1))./repmat(maxdx-mindx,totnum,1), ...
    (digitdata(:,5+n_basis_functions:4+2*n_basis_functions)-repmat(mindy, totnum, 1))./repmat(maxdy-mindy, totnum,1)
    ];
batchtargets1 = targets;
% for b=1:numbatches
%   batchdata(:,:,b) = digitdata(randomorder(1+(b-1)*batchsize:b*batchsize), :);
%   batchtargets(:,:,b) = targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
% end;
rr=randperm(totnum);
% rr=[1:20:totnum 2:20:totnum 3:20:totnum 4:20:totnum 5:20:totnum...
%     6:20:totnum 7:20:totnum 8:20:totnum 9:20:totnum 10:20:totnum...
%     11:20:totnum 12:20:totnum 13:20:totnum 14:20:totnum 15:20:totnum...
%     16:20:totnum 17:20:totnum 18:20:totnum 19:20:totnum 20:20:totnum];
% rr=[1:21:totnum 2:21:totnum 3:21:totnum 4:21:totnum 5:21:totnum...
%     6:21:totnum 7:21:totnum 8:21:totnum 9:21:totnum 10:21:totnum...
%     11:21:totnum 12:21:totnum 13:21:totnum 14:21:totnum 15:21:totnum...
%     16:21:totnum 17:21:totnum 18:21:totnum 19:21:totnum 20:21:totnum 21:21:totnum];
%rr=1:totnum;
batchdata1=repmat(batchdata1(rr,:),1,nRepeat);
batchtargets1=batchtargets1(rr,:);

for iBatch=1:numbatches
    batchdata(:,:,iBatch)=batchdata1((iBatch-1)*batchsize+1:iBatch*batchsize,:);
    batchtargets(:,:,iBatch)=batchtargets1((iBatch-1)*batchsize+1:iBatch*batchsize,:);
    
end
%%% Reset random seeds
rand('state',sum(100*clock));
randn('state',sum(100*clock));
save temp_batch_dmp
save maxmin maxdx mindx maxdy mindy n_basis_functions nRepeat

%% check the data batches
fig=figure('name',num2str(n_basis_functions))
show_rbm(batchdata(randi(totnum,50,1),:,1),numdims)
% for iSample=1:50%batchsize
%     coordsX1=batchdat a(iSample,1,1);
%     coordsY1=batchdata(iSample,2,1);
%     coordsX2=batchdata(iSample,3,1);
%     coordsY2=batchdata(iSample,4,1);
%         
%     visible= batchdata(iSample,5:end,1);
%     visible = visible + 0.0*rand(size(visible)); %testing stability
%     visible = visible.*[(maxdx-mindx)*ones(1,n_basis_functions) (maxdy-mindy)*ones(1,n_basis_functions)] ...
%         +[mindx*ones(1,n_basis_functions) mindy*ones(1,n_basis_functions)];
%     visible = reshape(visible,2,n_basis_functions);
%     subplot(5, nTargets,iSample)
%     iDigit=find(batchtargets(iSample,:,1)>0);
%     [ trajectory ] = dmpintegrate([ coordsX1 coordsY1],...
%         [coordsX2 coordsY2 ],visible,time,dt,time_exec,order);
%     hold on
%     handle = plot(trajectory.y(:,1,1),trajectory.y(:,2,1));
%     title(find(batchtargets(iSample,:,1)>0))
% end

%% create the dbn
maxepoch=1000;
numdims  =  size(batchdata,2);
numhid=500; numpen=1000;
nodes = [numdims numhid numpen];
dbn = randDBN(nodes,nTargets,'ASSOC');

% train DBM
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