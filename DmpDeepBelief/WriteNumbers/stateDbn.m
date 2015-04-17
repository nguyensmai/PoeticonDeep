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
% batchdata -- the data that is divided into batches (numcases ~ numbatches)
% restart   -- set to 1 if learning starts from beginning
function [dbn,errList] = stateDbn(batchdata,batchtargets,dbn,maxepoch,restart)

epsilonwv1     = 0.05;   % Learning rate for weights between the top and the hidden
epsilonwl1     = 0.05;   % Learning rate for weights between the top and the labels
epsilontb1     = 0.05;   % Learning rate for biases of top units
epsilonhb1     = 0.05;   % Learning rate for biases of hidden units
epsilonlb1     = 0.05;   % Learning rate for biases of label units
weightcost  = 0.001;
initialmomentum  = 0.5;
finalmomentum    = 0.9;
[numcases, ~, numbatches]=size(batchdata);
numdims=dbn.nodes(end-1);
nTargets= length(dbn.rbm{end}.labbiases);
numtop = length(dbn.rbm{end}.topbiases);
nLayers=length(dbn.rbm);
errList=[];

if restart ==1,
    % Initializing symmetric weights and biases.
    dbn.rbm{end} = assocRBM(length(dbn.rbm{end}.hidbiases), length(dbn.rbm{end}.topbiases), length(dbn.rbm{end}.labbiases));
end


% Initializing symmetric weights and biases.
hidtop    = dbn.rbm{end}.hidtop;
hidbiases = dbn.rbm{end}.hidbiases;

labtop    = dbn.rbm{end}.labtop;
labbiases = dbn.rbm{end}.labbiases;

topbiases = dbn.rbm{end}.topbiases;



postopprobs = zeros(numcases,numtop);
negtopprobs = zeros(numcases,numtop);
posprods    = zeros(numdims,numtop);
negprods    = zeros(numdims,numtop);
hidtopinc   = zeros(numdims,numtop);
labtopinc   = zeros(nTargets,numtop);
topbiasinc  = zeros(1,numtop);
hidbiasinc  = zeros(1,numdims);
labbiasinc  = zeros(1,nTargets);
fig2= figure('name','toprbm');
title('toprbm error');
hold on

%%fprintf(1,'epoch %d\r',epoch);
%fprintf(1,'epoch %d batch %d\r',epoch,batch);
  numCDiters = 1; %min([ceil(sqrt(epoch)) numCDitersMax]);
    time = epoch;%-(numCDiters-1)^2;
    if time>5,
        momentum=finalmomentum;
        epsilonwv     = epsilonwv1/(numCDiters*log2(epoch));   % Learning rate for weights between the top and the hidden
        epsilonwl     = epsilonwl1/(numCDiters*log2(epoch));   % Learning rate for weights between the top and the labels
        epsilontb     = epsilontb1/(numCDiters*log2(epoch));   % Learning rate for biases of top units
        epsilonhb     = epsilonhb1/(numCDiters*log2(epoch));   % Learning rate for biases of hidden units
        epsilonlb     = epsilonlb1/(numCDiters*log2(epoch));   % Learning rate for biases of label units
        
    else
        momentum=initialmomentum;
        epsilonwv     = epsilonwv1/(1.0*numCDiters);   % Learning rate for weights between the top and the hidden
        epsilonwl     = epsilonwl1/(1.0*numCDiters);   % Learning rate for weights between the top and the labels
        epsilontb     = epsilontb1/(1.0*numCDiters);   % Learning rate for biases of top units
        epsilonhb     = epsilonhb1/(1.0*numCDiters);   % Learning rate for biases of hidden units
        epsilonlb     = epsilonlb1/(1.0*numCDiters);   % Learning rate for biases of label units
    end;
    
    
    for batch = 1:numbatches,
        
        posstates{1}  = batchdata(:,:,batch);
        targets=batchtargets(:,:,batch);
        
        %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %1.1 First  RBM
        if( strcmpi( 'GBRBM', dbn.rbm{1}.type ) ) % if bernoulli
            posstates{1}= posstates{1}./repmat(exp(dbn.rbm{1}.z),numcases,1);
        end
        
        i=1;
        posprobs{i+1} = 1./(1 + exp( ...
            -2*posstates{i}*dbn.rbm{i}.vishid...
            - 2*repmat(dbn.rbm{i}.hidbiases,numcases,1)...
            ));
        posstates{i+1} = posprobs{i+1} > rand(size(posprobs{i+1}));
        
        
        %1.2. Sigmoid Belief Networks
        for i=1:nLayers - 1
            posprobs{i+1} = 1./(1 + exp( ...
                -posstates{i}*dbn.rbm{i}.vishid...
                -repmat(dbn.rbm{i}.hidbiases,numcases,1)...
                ));
            posstates{i+1} = posprobs{i+1} > rand(size(posprobs{i+1}));
        end
        
        %1.3. top RBM
        i=nLayers;
        posprobs{i+1} = 1./(1 + exp( ...
            -posstates{i}*dbn.rbm{i}.hidtop ...
            -targets*dbn.rbm{i}.labtop ...
            -repmat(dbn.rbm{i}.topbiases,numcases,1)...
            ));
        posstates{i+1} = posprobs{i+1} > rand(size(posprobs{i+1}));
        
        posprods = double(posstates{i})' * posprobs{i+1}/numcases;
        poslabprods = double(targets)' * posprobs{i+1}/numcases;
        
        poshidact = mean(posstates{i});
        poslabact = mean(targets);
        postopact = mean(postopprobs);

        %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        postopprobs_temp=postopprobs;
        
        %%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for iter=1:numCDiters
            
            postopstates = postopprobs_temp > rand(numcases,numtop);
            
            negdataprobs  = 1./(1 + exp(-postopstates*(2*hidtop)' - repmat(2*hidbiases,numcases,1)));
            negdata = negdataprobs > rand(size(negdataprobs));
            % negdata =  (postopstates*hidtop') + repmat(hidbiases,numcases,1);
            %             neglabel = exp(postopstates*labtop'+repmat(labbiases,numcases,1));
            %             neglabel =
            %             neglabel./repmat(sum(neglabel,2),1,nTargets);
            neglabprobs = exp(postopstates*labtop'+repmat(labbiases,numcases,1));
            neglabprobs = neglabprobs./(sum(neglabprobs,2)*ones(1,nTargets));
            xx = cumsum(neglabprobs,2);
            xx1 = rand(numcases,1);
            neglabel = neglabprobs*0;
            for jj=1:numcases
                index = min(find(xx1(jj) <= xx(jj,:)));
                neglabel(jj,index) = 1;
            end
            
            postopprobs_temp = 1./(1 + exp(-negdata*hidtop -neglabel*labtop - repmat(topbiases,numcases,1)));
            %negtopstates = negtopprobs>rand(numcases, numtop);
        end
        postopprobs= postopprobs_temp;
        
        negtopact = mean(negtopprobs);
        
        negprods  = double(negdata')*double(negtopprobs)/numcases;
        neghidact = mean(negdata);
        
        neglabprods  = neglabel'*negtopprobs/numcases;
        neglabact = mean(neglabel);
        %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      
    
    batchposprobs(:,:,batch)=posprobs;
    batchposstates(:,:,batch)=posstates;
    batchnegprobs(:,:,batch)=negprobs;
    batchnegstates(:,:,batch)=negstates;
    end
end
