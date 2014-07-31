P = path();
path(P,'../../dmp_bbo_matlab_deprecated-master_deprecated/dynamicmovementprimitive')
path(P,'../../Autoencoder_Code')
digitdata=[];
targets=[];

maxepoch=2000; 
numhid=500; numpen=1000; numpen2=1000; 


%% display the numbers of the data
for iDigit=1:9
    for iSample=1:nSamples(iDigit)
        
        trajectory.t = ( 0:1:( numel(data{iDigit,iSample}{1})-1) )';
        trajectory.y = [data{iDigit,iSample}{1}(:),data{iDigit,iSample}{2}(:)];
        yd = trajectory.y(2:end,:)-trajectory.y(1:end-1,:);
        d= size(yd,2);
        trajectory.yd= [zeros(1,d); yd];
        
        ydd = trajectory.yd(2:end,:)-trajectory.yd(1:end-1,:);
        trajectory.ydd = [zeros(1,d);ydd];
        
        
        order = 3;
        n_basis_functions = 100;
        subplot(9,max(nSamples), (iDigit-1)*max(nSamples) +iSample)
        [ theta y0 g0 ] = dmptrain(trajectory,order,n_basis_functions);
        digitdata = [digitdata; theta(:)'];
        target = zeros(1,9);
        target(iDigit) = 1;
        targets = [targets; target]; 
        hold on
        plot( data{iDigit,iSample}{1},data{iDigit,iSample}{2},'-r')
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


%% train each layer with rbm

maxepoch=10000; 
fprintf(1,'Pretraining Layer 1 with RBM: %d-%d \n',numdims,numhid);
restart=1;
rbmgaussian;
hidrecbiases=hidbiases; 
save mnistvhclassify vishid hidrecbiases visbiases;

maxepoch=3000; 
fprintf(1,'\nPretraining Layer 2 with RBM: %d-%d \n',numhid,numpen);
batchdata=batchposhidprobs;
numhid=numpen;
restart=1;
rbm;
hidpen=vishid; penrecbiases=hidbiases; hidgenbiases=visbiases;
save mnisthpclassify hidpen penrecbiases hidgenbiases;

fprintf(1,'\nPretraining Layer 3 with RBM: %d-%d \n',numpen,numpen2);
batchdata=batchposhidprobs;
numhid=numpen2;
restart=1;
rbm;
hidpen2=vishid; penrecbiases2=hidbiases; hidgenbiases2=visbiases;
save mnisthp2classify hidpen2 penrecbiases2 hidgenbiases2;


%%
backpropclassifyHinton; 

plotWeights;
