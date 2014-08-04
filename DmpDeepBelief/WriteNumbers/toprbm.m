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
function [hidtop, labtop, topbiases, hidbiases, labbiases, restart] = toprbm(batchdata,batchtargets,numtop,nTargets,maxepoch,restart)

epsilonwv     = 0.0001;   % Learning rate for weights
epsilonwl     = 0.0001;   % Learning rate for weights
epsilonvb     = 0.0001;   % Learning rate for biases of visible units
epsilonhb     = 0.0001;   % Learning rate for biases of hidden units
epsilonlb     = 0.0001;   % Learning rate for biases of label units
weightcost  = 0.0002;
initialmomentum  = 0.1;
finalmomentum    = 0.5;
numCDiters =50;
[numcases numdims numbatches]=size(batchdata);

if restart ==1,
    restart=0;
    epoch=1;
    
    % Initializing symmetric weights and biases.
    hidtop     = 0.1*randn(numdims, numtop);
    hidbiases  = zeros(1,numdims);
    
    labtop     = 0.1*randn(nTargets,numtop);
    labbiases  = zeros(1,nTargets);
    
    topbiases  = zeros(1,numtop);
    
    
    
    postopprobs = zeros(numcases,numtop);
    negtopprobs = zeros(numcases,numtop);
    posprods    = zeros(numdims,numtop);
    negprods    = zeros(numdims,numtop);
    hidtopinc   = zeros(numdims,numtop);
    labtopinc   = zeros(nTargets,numtop);
    hidbiasinc  = zeros(1,numtop);
    hidbiasinc  = zeros(1,numdims);
    labbiasinc  = zeros(1,nTargets);
    batchpostopprobs=zeros(numcases,numtop,numbatches);
    figure
    title('toprbm');
end

%fprintf(1,'epoch %d\r',epoch);
for batch = 1:numbatches,
    %fprintf(1,'epoch %d batch %d\r',epoch,batch);
    for epoch = epoch:maxepoch
        
        data = batchdata(:,:,batch);
        targets = batchtargets(:,:,batch);
        errsum=0;
        
        for iter=1:numCDiters
            
            %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            postopprobs = 1./(1 + exp(-data*hidtop -targets*labtop - repmat(topbiases,numcases,1)));
            batchpostopprobs(:,:,batch)= postopprobs;
            
            postopact   = sum(postopprobs);
            
            posprods    = data' * postopprobs;
            posvisact = sum(data);
            
            poslabprods    = targets' *postopprobs;
            poslabact = sum(targets);
            %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            poshidstates = postopprobs > rand(numcases,numtop);
            
            %%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            negdataprobs  = 1./(1 + exp(-poshidstates*hidtop' - repmat(hidbiases,numcases,1)));
            negdata = negdataprobs > rand(numcases,numdims);
            % negdata =  (poshidstates*hidtop') + repmat(hidbiases,numcases,1);
            neglabel = exp(poshidstates*labtop'+repmat(labbiases,numcases,1));
            neglabel = neglabel./repmat(sum(neglabel,2),1,nTargets);
            negtopprobs = 1./(1 + exp(-negdata*hidtop -neglabel*labtop - repmat(topbiases,numcases,1)));
            negtopstates = negtopprobs>rand(numcases, numtop);
            
            neghidact = sum(negtopprobs);
            
            negprods  = double(negdata')*double(negtopstates);
            negvisact = sum(negdata);
            
            neglabprods  = neglabel'*negtopprobs;
            neglabact = sum(neglabel);
            %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            err1= sum(sum( (data-negdata).^2 ));
            err2= sum(sum( (targets-neglabel).^2 ));
            errsum = err1 +err2 + errsum;
            
            if epoch>5,
                momentum=finalmomentum;
            else
                momentum=initialmomentum;
            end;
            
            %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            hidtopinc = momentum*hidtopinc + ...
                epsilonwv*( (posprods-negprods)/numcases - weightcost*hidtop);
            labtopinc = momentum*labtopinc + ...
                epsilonwl*( (poslabprods-neglabprods)/numcases - weightcost*labtop);
            hidbiasinc = momentum*hidbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
            labbiasinc = momentum*labbiasinc + (epsilonlb/numcases)*(poslabact-neglabact);
            hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(postopact-neghidact);
            
            hidtop = hidtop + hidtopinc;
            hidbiases = hidbiases + hidbiasinc;
            topbiases = topbiases + hidbiasinc;
            labbiases = labbiases + labbiasinc;
            %%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            data = negdata;
        end
        fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum);
        plot(epoch, errsum,'x');
        hold on;
        plot(epoch, err1,'xg');
        plot(epoch, err2,'xr');
    end
end;
