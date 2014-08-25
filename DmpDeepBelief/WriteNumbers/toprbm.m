% Version 1.000
%
% Code provided by Geoff Hinton and Ruslan Salakhutdinov
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

% This program trains Restricted Boltzmann Machine in which
% visible, binary, stochastic pixels are connected to
% hidden, binary, stochastic feature detectors using symmetrically
% weighted connections. Learning is done with 1-step Contrastive Divergence.
% The program assumes that the following variables are set externally:
% maxepoch  -- maximum number of epochs
% numtop    -- number of hidden units
% batchdata -- the data that is divided into batches (numcases numdims numbatches)
% restart   -- set to 1 if learning starts from beginning
function [rbm,errList] = toprbm(batchdata,batchtargets,rbm,maxepoch,restart)

epsilonwv1     = 0.1;   % Learning rate for weights between the top and the hidden
epsilonwl1     = 0.1;   % Learning rate for weights between the top and the labels
epsilontb1     = 0.1;   % Learning rate for biases of top units
epsilonhb1     = 0.1;   % Learning rate for biases of hidden units
epsilonlb1     = 0.1;   % Learning rate for biases of label units
weightcost  = 0.02;
initialmomentum  = 0.5;
finalmomentum    = 0.9;
[numcases numdims numbatches]=size(batchdata);
nTargets= length(rbm.labbiases);
numtop = length(rbm.topbiases);
errList=[];

if restart ==1,
    % Initializing symmetric weights and biases.
    rbm = assocRBM(length(rbm.hidbiases), length(rbm.topbiases), length(rbm.labbiases));
end


% Initializing symmetric weights and biases.
hidtop    = rbm.hidtop;
hidbiases = rbm.hidbiases;

labtop    = rbm.labtop;
labbiases = rbm.labbiases;

topbiases = rbm.topbiases;



postopprobs = zeros(numcases,numtop);
negtopprobs = zeros(numcases,numtop);
posprods    = zeros(numdims,numtop);
negprods    = zeros(numdims,numtop);
hidtopinc   = zeros(numdims,numtop);
labtopinc   = zeros(nTargets,numtop);
topbiasinc  = zeros(1,numtop);
hidbiasinc  = zeros(1,numdims);
labbiasinc  = zeros(1,nTargets);
figure('name','toprbm')
title('toprbm error');
hold on

%%fprintf(1,'epoch %d\r',epoch);
%fprintf(1,'epoch %d batch %d\r',epoch,batch);
for epoch = 1:maxepoch
    errsum=0;
    numCDiters = ceil(epoch/20); %min([ceil(sqrt(epoch)) numCDitersMax]);
    time = epoch;%-(numCDiters-1)^2;
    if time>50,
        momentum=finalmomentum;
        epsilonwv     = epsilonwv1/(numCDiters*log2(epoch));   % Learning rate for weights between the top and the hidden
        epsilonwl     = epsilonwl1/(numCDiters*log2(epoch));   % Learning rate for weights between the top and the labels
        epsilontb     = epsilontb1/(numCDiters*log2(epoch));   % Learning rate for biases of top units
        epsilonhb     = epsilonhb1/(numCDiters*log2(epoch));   % Learning rate for biases of hidden units
        epsilonlb     = epsilonlb1/(numCDiters*log2(epoch));   % Learning rate for biases of label units
        
    else
        momentum=initialmomentum;
        epsilonwv     = epsilonwv1/numCDiters;   % Learning rate for weights between the top and the hidden
        epsilonwl     = epsilonwl1/numCDiters;   % Learning rate for weights between the top and the labels
        epsilontb     = epsilontb1/numCDiters;   % Learning rate for biases of top units
        epsilonhb     = epsilonhb1/numCDiters;   % Learning rate for biases of hidden units
        epsilonlb     = epsilonlb1/numCDiters;   % Learning rate for biases of label units
    end;
    
    
    for batch = 1:numbatches,
        
        data = batchdata(:,:,batch);
        targets = batchtargets(:,:,batch);
        
        %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        postopprobs = 1./(1 + exp(-data*hidtop -targets*labtop - repmat(topbiases,numcases,1)));
        
        postopact   = mean(postopprobs);
        
        posprods    = data' * postopprobs/numcases;
        poshidact = mean(data);
        
        poslabprods    = targets' * postopprobs/numcases;
        poslabact = mean(targets);
        %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        postopprobs_temp=postopprobs;
        
        %%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for iter=1:numCDiters
            
            postopstates = postopprobs_temp > rand(numcases,numtop);
            
            negdataprobs  = 1./(1 + exp(-postopstates*hidtop' - repmat(hidbiases,numcases,1)));
            negdata = negdataprobs > rand(numcases,numdims);
            % negdata =  (postopstates*hidtop') + repmat(hidbiases,numcases,1);
            %             neglabel = exp(postopstates*labtop'+repmat(labbiases,numcases,1));
            %             neglabel = neglabel./repmat(sum(neglabel,2),1,nTargets);
            neglabprobs = exp(postopstates*labtop'+repmat(labbiases,numcases,1));
            neglabprobs = neglabprobs./(sum(neglabprobs,2)*ones(1,nTargets));
            xx = cumsum(neglabprobs,2);
            xx1 = rand(numcases,1);
            neglabel = neglabprobs*0;
            for jj=1:numcases
                index = min(find(xx1(jj) <= xx(jj,:)));
                neglabel(jj,index) = 1;
            end
            
            negtopprobs = 1./(1 + exp(-negdata*hidtop -neglabel*labtop - repmat(topbiases,numcases,1)));
            %negtopstates = negtopprobs>rand(numcases, numtop);
            postopprobs_temp = negtopprobs;
        end
        
        negtopact = mean(negtopprobs);
        
        negprods  = double(negdata')*double(negtopprobs)/numcases;
        neghidact = mean(negdata);
        
        neglabprods  = neglabel'*negtopprobs/numcases;
        neglabact = mean(neglabel);
        %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        errD = sum(sum( (data-negdata).^2 ))/(numcases*numdims);
        errL = sum(sum( (targets-neglabel).^2 ))/(numcases*nTargets);
        errsum = errD +errL + errsum;
        
        
        
        
        %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        hidtopinc = momentum*hidtopinc + ...
            epsilonwv*( (posprods-negprods) - weightcost*hidtop);
        labtopinc = momentum*labtopinc + ...
            epsilonwl*( (poslabprods-neglabprods) - weightcost*labtop);
        hidbiasinc = momentum*hidbiasinc + (epsilonhb)*(poshidact-neghidact);
        labbiasinc = momentum*labbiasinc + (epsilonlb)*(poslabact-neglabact);
        topbiasinc = momentum*topbiasinc + (epsilontb)*(postopact-negtopact);
        
        hidtop = hidtop + hidtopinc;
        hidbiases = hidbiases + hidbiasinc;
        topbiases = topbiases + topbiasinc;
        labbiases = labbiases + labbiasinc;
        %%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    errList=[errList; errsum errD errL];
    
    if mod(epoch,100)==1
        fprintf(1, 'TOPRBM : epoch %4i error %6.6f\n', epoch, errsum);
        plot(epoch, errsum,'x');
        hold on;
        plot(epoch, errD,'xg');
        plot(epoch, errL,'xr');
        drawnow
        save layerTOPRBM
    end
    
end

rbm.hidtop=hidtop;
rbm.hidbiases=hidbiases;
rbm.labtop=labtop;
rbm.labbiases=labbiases;
rbm.topbiases=topbiases;
end
