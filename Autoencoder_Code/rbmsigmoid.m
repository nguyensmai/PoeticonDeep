% Version 1.000
%
% Code provided by Geoff Hinton and Ruslan Salakhutdinov 
% then edited by Sao Mai Nguyen
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

function [rbm,batchposhidprobs,errL, batchnegdata] = rbmsigmoid(batchdata,rbm,maxepoch, restart)

epsilonw      = 10^-2;   % Learning rate for weights
epsilonvb     = 10^-2;   % Learning rate for biases of visible units
epsilonhb     = 10^-2;   % Learning rate for biases of hidden units
% epsilonw      = 10^-6;   % Learning rate for weights
% epsilonvb     = 10^-6;   % Learning rate for biases of visible units
% epsilonhb     = 10^-6;   % Learning rate for biases of hidden units
weightcost    = 0.0;
initialmomentum  = 0.5;
finalmomentum    = 0.8;

[numcases numdims numbatches]=size(batchdata);
numhid = length(rbm.hidbiases);
epoch=1;
errL=[];

if restart ==1,
    % Initializing symmetric weights and biases.
    rbm= randRBM(length(rbm.visbiases), length(rbm.hidbiases), rbm.type);
end

vishid     = rbm.vishid;
hidbiases  = rbm.hidbiases;
visbiases  = rbm.visbiases;

poshidprobs = zeros(numcases,numhid);
neghidprobs = zeros(numcases,numhid);
posprods    = zeros(numdims,numhid);
negprods    = zeros(numdims,numhid);
vishidinc  = zeros(numdims,numhid);
hidbiasinc = zeros(1,numhid);
visbiasinc = zeros(1,numdims);
batchposhidprobs=zeros(numcases,numhid,numbatches);
batchposhidstates=zeros(numcases,numhid,numbatches);
batchnegdata=zeros(size(batchdata));
fig1= figure;
fig2= figure;

momentum=initialmomentum;
epoch=1;
%%
for epoch = epoch:maxepoch,
%while errsum>0.1
    % fprintf(1,'epoch %d\r',epoch);
    errsum=0;
    epsilonw = max(epsilonw/1.0001,10^-9);
    epsilonvb = max(epsilonvb/1.0001,10^-9);
    epsilonhb = max(epsilonhb/1.0001,10^-9);
    %momentum=min(momentum*1.00001,0.99);
    rr=randperm(numbatches);
    if epoch>10^2
        momentum=0.99;
    end
    
    %epoch=epoch+1;
    for batch = rr,
        %fprintf(1,'epoch %d batch %d\r',epoch,batch);
        
        %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        dataprobs = batchdata(:,:,batch);
        data= 0.0+(dataprobs>rand(size(dataprobs)));
        poshidprobs = 1./(1 + exp(-data*(2*vishid) - repmat(2*hidbiases,numcases,1)));
        batchposhidprobs(:,:,batch)=poshidprobs;
        posprods    = data' * poshidprobs/numcases;
        poshidact   = mean(poshidprobs);
        posvisact = mean(data);
        
        %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        poshidstates = poshidprobs > rand(numcases,numhid);
        batchposhidstates(:,:,batch)= poshidstates;
        
        %%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        negdata = 1./(1 + exp(-poshidstates*vishid' - repmat(visbiases,numcases,1)));
%         negdata = negdataprobs > rand(numcases,numdims);
        neghidprobs = 1./(1 + exp(-negdata*(2*vishid) - repmat(2*hidbiases,numcases,1)));
        negprods  = negdata'*neghidprobs/numcases;
        neghidact = mean(neghidprobs);
        negvisact = mean(negdata);
        batchnegdata(:,:,batch)=negdata;
        
        %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        err= sum(sum( (dataprobs-negdata).^2 ))/(numcases*numdims);
        errsum = err + errsum;
        
        %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


        
        vishidinc = momentum*vishidinc + ...
            epsilonw*( (posprods-negprods) - weightcost*vishid);
        visbiasinc = momentum*visbiasinc + (epsilonvb)*(posvisact-negvisact);
        hidbiasinc = momentum*hidbiasinc + (epsilonhb)*(poshidact-neghidact);
        
        vishid    = vishid + vishidinc;
        visbiases = visbiases + visbiasinc;
        hidbiases = hidbiases + hidbiasinc;
        
        %%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    end
    
    errsum=errsum/numbatches;    
    errL=[errL;errsum];

    if mod(epoch,100)==1
        fprintf(1, 'RBMSIGMOID : epoch %4i error %6.6f\n', epoch, errsum);
        figure(fig1)
        plot(epoch, errsum,'x');
        hold on;
        drawnow;
        save layerSIGMOID
        figure(fig2)
        clf
        show_rbm(batchnegdata(1:50,:,1),numdims)
        drawnow
        title(['RBMSIGMOID 1: epoch ', num2str(epoch), ' error ',num2str(errsum)])
    end
end;

rbm.vishid    = vishid;
rbm.hidbiases = hidbiases;
rbm.visbiases = visbiases;

end
