addpath('../WriteNumbers');
%addpath('../../Autoencoder_Code');

%%
% global report_calls_to_sample_bernoulli
% report_calls_to_sample_bernoulli = false;
% global data_sets
% a4_init;
% batchdata = data_sets.training.inputs';
% batchtargets = data_sets.training.targets';
load batches_dbm.mat
nTargets = 10;
numdims=size(batchdata,2);


%% create the dbn
numhid=500; numpen=1000; %numpen2=100;
nodes = [numdims numhid numpen];% numpen2];
dbn = randDBN(nodes,nTargets,'ASSOC');
show_rbm(batchdata(1:81,:,end),numdims)
title('input data')

%% train each rbm with contrastive-divergence

maxepoch=51;
fprintf(1,'Pretraining Layer 1 with RBM: %d-%d \n',numdims,numhid);
restart=1;
[dbn.rbm{1},batchposhidprobs, errL1, negdata] = rbmsigmoid(batchdata,dbn.rbm{1},maxepoch,restart);
title('error for layer1');
save layer1
show_rbm(negdata(1:81,:,end),numdims)
title('negdata: reconstruction');

%%
% maxepoch=1;
% fprintf(1,'\nPretraining Layer 2 with RBM: %d-%d \n',numhid,numpen);
% restart=0;
% [dbn.rbm{2}, batchpospenprobs,errL2] = rbmsigmoid(poshidstates,dbn.rbm{2},maxepoch, restart);
% title('layer2');
% save layer2;

maxepoch=41;
%fprintf(1,'\nPretraining Layer 3  (hidden and labels) with RBM: [%d %d]-%d \n',numpen,nTargets,numpen2);
restart=1;
[dbn,errL3] = toprbm(batchdata,batchtargets,dbn,maxepoch,restart);
title('error for layer3');
save layer3;



%% mean-field
dbn = untie(dbn);
maxepoch=151;
restart=0;
[dbn, errBack, negdata_CD1, negstates, neglabels] = dbm_mf(batchdata,batchtargets,dbn, maxepoch,restart);
title('error for mean-field');
save dbmMfTrain
figure
show_rbm(negdata_CD1(1:81,:,end),numdims)
title('negstates: reconstruction after dbm_mf');


%%
figure
[dbn,  negstates] = generativeModel(batchdata,batchtargets,eye(nTargets),dbn,1)
save layergenerative;

figure
[~, errTest, negdata_CD1, negstates, neglabels] = dbm_mf(batchdata,batchtargets,dbn, 1,0);
save finaltest;

