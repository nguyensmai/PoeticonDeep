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
% numhid    -- number of hidden units
% batchdata -- the data that is divided into batches (numcases numdims numbatches)
% restart   -- set to 1 if learning starts from beginning
function [rbm,batchposhidprobs, errL, negdata] = rbmgaussian(batchdata,rbm,maxepoch, restart)

epsilonw1      = 0.1;   % Learning rate for weights
epsilonvb1     = 0.1;   % Learning rate for biases of visible units
epsilonhb1     = 0.1;   % Learning rate for biases of hidden units
epsilonz1     = 0.1;   % Learning rate for biases of hidden units
weightcost  = 0.2;
initialmomentum  = 0.5;
finalmomentum    = 0.9;

[numcases numdims numbatches]=size(batchdata);
numhid = length(rbm.hidbiases);
errL=[];
epoch=1;

if restart ==1,
    % Initializing symmetric weights and biases.
    rbm = randRBM(length(rbm.visbiases), length(rbm.hidbiases), rbm.type);
end

vishid     = rbm.vishid;
hidbiases  = rbm.hidbiases;
visbiases  = rbm.visbiases;
z          = rbm.z;

% poshidprobs = zeros(numcases,numhid);
% neghidprobs = zeros(numcases,numhid);
% posprods    = zeros(numdims,numhid);
% negprods    = zeros(numdims,numhid);
vishidinc   = zeros(numdims,numhid);
hidbiasinc  = zeros(1,numhid);
visbiasinc  = zeros(1,numdims);
batchposhidprobs=zeros(numcases,numhid,numbatches);
batchposhidstates=zeros(numcases,numhid,numbatches);
figure;

errsum=20;
epoch=0;
for epoch = epoch:maxepoch,
%while errsum>0.1
    % fprintf(1,'epoch %d\r',epoch);
    errsum=0;
   % epoch=epoch+1;
    for batch = 1:numbatches,
        %fprintf(1,'epoch %d batch %d\r',epoch,batch);
        
        %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        data = batchdata(:,:,batch);
        dataz= data./repmat(exp(z),numcases,1);
        poshidprobs = 1./(1 + exp(-dataz*vishid - repmat(hidbiases,numcases,1)));
        batchposhidprobs(:,:,batch)=poshidprobs;
        posprods    = data' * poshidprobs/numcases;
        poshidact   = mean(poshidprobs);
        posvisact = mean(data);
        
        %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        poshidstates = poshidprobs > rand(numcases,numhid);
        batchposhidstates(:,:,batch)= poshidstates;
 
        %%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %  negmu    = 1./(1 + exp(-poshidstates*vishid' - repmat(visbiases,numcases,1)));
        negmu    = poshidstates*vishid' + repmat(visbiases,numcases,1);
        std = repmat(sqrt(exp(z)), numcases,1);
        negdata  = random('norm', negmu, std);
        negdataz  = negdata./repmat(exp(z),numcases,1); 
        neghidprobs = 1./(1 + exp(-negdataz*vishid - repmat(hidbiases,numcases,1)));
        negprods  = negdata'*neghidprobs/numcases;
        neghidact = mean(neghidprobs);
        negvisact = mean(negdata);
        
        %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        err = sum(sum( (data-negdata).^2 ))/(numcases*numdims);
        errsum = errsum + err;
        
        if epoch>500,
            momentum=0.99;
            epsilonw      = epsilonw1/sqrt(epoch);
            epsilonvb     = epsilonvb1/sqrt(epoch);
            epsilonhb     = epsilonhb1/sqrt(epoch);
            epsilonz     = epsilonz1/sqrt(epoch);
        elseif epoch>50,
            momentum=finalmomentum;
            epsilonw      = epsilonw1/sqrt(epoch);
            epsilonvb     = epsilonvb1/sqrt(epoch);
            epsilonhb     = epsilonhb1/sqrt(epoch);
            epsilonz     = epsilonz1/sqrt(epoch);
        else
            momentum=initialmomentum;
            epsilonw      = epsilonw1;
            epsilonvb     = epsilonvb1;
            epsilonhb     = epsilonhb1;
            epsilonz     = epsilonz1;
        end;
        
        %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        vishidinc = momentum*vishidinc + ...
            epsilonw*( (posprods-negprods)./(repmat(exp(z)',1,numhid)) - weightcost*vishid);
        visbiasinc = momentum*visbiasinc + epsilonvb./(exp(z)).*(posvisact-negvisact);
        hidbiasinc = momentum*hidbiasinc + epsilonhb*(poshidact-neghidact);
        zinc = epsilonz * exp(-z) .*(...
            mean(1/2*(data-repmat(visbiases,numcases,1)).^2 - data.*(poshidprobs*vishid'),1) ...
            - mean(1/2*(negdata-repmat(visbiases,numcases,1)).^2 - negdata.*(neghidprobs*vishid'),1) ...
            );
        
        vishid = vishid + vishidinc;
        visbiases = visbiases + visbiasinc;
        hidbiases = hidbiases + hidbiasinc;
        z = z+zinc;
        
        %%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    end
    errsum=errsum/numbatches;
    errL=[errL;errsum];

    if mod(epoch,10)==0
        
        fprintf(1, 'RBMGAUSSIAN: epoch %4i error %6.6f  \n', epoch, errsum);
        plot(epoch, errsum,'x');
        hold on;
        drawnow
        save layerGAUSSIAN
    end
    
end;

rbm.vishid    = vishid;
rbm.hidbiases = hidbiases;
rbm.visbiases = visbiases;
rbm.z         = z;

%plotReconstruction(negdata);
%set(gcf,'name','rbm gaussian')
end



