% Version 1.000
%
% Code provided by Ruslan Salakhutdinov
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

function [dbn,  negstates] = generativeModel(batchdata,batchtargets,labels,dbn, maxepoch)




%% initialisation
epsilonw      = 0.001;   % Learning rate for weights
epsilonvb     = 0.001;   % Learning rate for biases of visible units
epsilonhb     = 0.001;   % Learning rate for biases of hidden units
weightcost  = 0.0002;
initialmomentum  = 0.5;
finalmomentum    = 0.9;

[numcases numdims numbatches]=size(batchdata);

nTargets=10;
dbn.nodes(1)=numdims;

restart=0;
epoch=1;


vishidinc  = zeros(dbn.nodes(1),dbn.nodes(2));
hidtopinc  = zeros(dbn.nodes(2),dbn.nodes(3));
labtopinc =  zeros(nTargets,dbn.nodes(3));



hidbiasinc = zeros(1,dbn.nodes(2));
visbiasinc = zeros(1,dbn.nodes(1));
penbiasinc = zeros(1,dbn.nodes(3));
labbiasinc = zeros(1,nTargets);

%%%% This code also adds sparcity penalty
sparsetarget = .2;
sparsetarget2 = .1;
sparsecost = .001;
sparsedamping = .9;

hidmeans = sparsetarget*ones(1,dbn.nodes(2));
dbn.rbm{2}.topbiases  = 0*log(sparsetarget2/(1-sparsetarget2))*ones(1,dbn.nodes(end));
penmeans = sparsetarget2*ones(1,dbn.nodes(end));


hidbiases = (dbn.rbm{1}.hidbiases + dbn.rbm{2}.hidbiases);
epoch=1;

neglabstates = 1/nTargets*(ones(numcases,nTargets));
data = round(rand(100,numdims));
negprobs{2} = 1./(1 + exp(-data*(2*dbn.rbm{1}.vishid) - repmat(hidbiases,numcases,1)));


epsilonw      = epsilonw/(1.000015^((epoch-1)*600));
epsilonvb      = epsilonvb/(1.000015^((epoch-1)*600));
epsilonhb      = epsilonhb/(1.000015^((epoch-1)*600));

errMF=[];
nLayers= numel(dbn.rbm);

%% beginning of learning
for epoch = epoch:maxepoch
    [numcases numdims numbatches]=size(batchdata);
    
    fprintf(1,'epoch %d \t eps %f\r',epoch,epsilonw);
    errsum=0;
    
    
    counter=0;
    %rr = randperm(numbatches);
    %batch=0;
    %for batch_rr = rr; %1:numbatches,
    for batch=1:numbatches
       % fprintf(1,'epoch %d batch %d\r',epoch,batch);
        epsilonw = max(epsilonw/1.000015,0.00010);
        epsilonvb = max(epsilonvb/1.000015,0.00010);
        epsilonhb = max(epsilonhb/1.000015,0.00010);
        
        
        %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        data = batchdata(:,:,batch);
        targets = batchtargets(:,:,batch);
        posprobs{1}=data;
        data = double(data > rand(numcases,dbn.nodes(1)));
        
        %%%%% First  MF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [posprobs{2}, posprobs{3}] = ...
            mf(data,targets,...
            dbn.rbm{1}.vishid,...
            hidbiases,...
            dbn.rbm{1}.visbiases,...
            dbn.rbm{2}.hidtop,...
            dbn.rbm{2}.topbiases,...
            dbn.rbm{2}.labtop,...
            dbn.rbm{1}.hidbiases);
        
        
        bias_hid= repmat(hidbiases,numcases,1);
        bias_pen = repmat(dbn.rbm{2}.topbiases,numcases,1);
        bias_vis = repmat(dbn.rbm{1}.visbiases,numcases,1);
        bias_lab = repmat(dbn.rbm{2}.labbiases,numcases,1);
        
        for i=1:nLayers
            posprods{i}    = posprobs{i}' * posprobs{i+1};
        end
        poslabprods = targets'*posprobs{nLayers+1};
        
        for i=1:nLayers+1
            posact{i}   = sum(posprobs{i});
        end
        poslabact   = sum(targets);
        
        
        %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        negdata_CD1 = 1./(1 + exp(-posprobs{2}*dbn.rbm{1}.vishid' - bias_vis));
        totin =  bias_lab + posprobs{3}*dbn.rbm{2}.labtop';
        poslabprobs1 = exp(totin);
        targetout = poslabprobs1./(sum(poslabprobs1,2)*ones(1,nTargets));
        [I J]=max(targetout,[],2);
        [I1 J1]=max(targets,[],2);
        counter=counter+length(find(J==J1));
        
        
        
        %%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for iter=1:5
            neglabstates = repmat(labels,10,1);
            negstates{2} = negprobs{2} > rand(numcases,dbn.nodes(2));
            
            negprobs{3} = 1./(1 + exp(-negstates{2}*dbn.rbm{2}.hidtop - neglabstates*dbn.rbm{2}.labtop - bias_pen));
            negstates{3} = negprobs{3} > rand(numcases,dbn.nodes(3));
            
            negprobs{1} = 1./(1 + exp(-negstates{2}*dbn.rbm{1}.vishid' - bias_vis));
            negstates{1} = negprobs{1} > rand(numcases,dbn.nodes(1));
            
            totin = negstates{3}*dbn.rbm{2}.labtop' + bias_lab;
            neglabprobs = exp(totin);
            neglabprobs = neglabprobs./(sum(neglabprobs,2)*ones(1,nTargets));
            
            xx = cumsum(neglabprobs,2);
            xx1 = rand(numcases,1);
            neglabstates = neglabstates*0;
            for jj=1:numcases
                index = min(find(xx1(jj) <= xx(jj,:)));
                neglabstates(jj,index) = 1;
            end
            xxx = sum(sum(neglabstates)) ;
            
            totin = negstates{1}*dbn.rbm{1}.vishid + bias_hid + negstates{3}*dbn.rbm{2}.hidtop';
            negprobs{2} = 1./(1 + exp(-totin));
            
        end
        negprobs{3} = 1./(1 + exp(-negprobs{2}*dbn.rbm{2}.hidtop - neglabprobs*dbn.rbm{2}.labtop - bias_pen));
        
        negprods{1}  = negstates{1}'*negprobs{2};
        negprods{2} = negprobs{2}'*negprobs{3};
        negact{2} = sum(negprobs{2});
        negact{3}   = sum(negprobs{3});
        negact{1} = sum(negstates{1});
        neglabact = sum(neglabstates);
        negprodslabpen = neglabstates'*negprobs{3};
        
        
        %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       % err= sum(sum( (data-negdata_CD1).^2 ));
        err= sum(sum( (data-negstates{1}).^2 ));
        errsum = err + errsum;
        
        if epoch >5
            momentum=finalmomentum;
        else
            momentum=initialmomentum;
        end;
        
        %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posact{1}-negact{1});
        labbiasinc = momentum*labbiasinc + (epsilonvb/numcases)*(poslabact-neglabact);
        
        hidmeans = sparsedamping*hidmeans + (1-sparsedamping)*posact{2}/numcases;
        sparsegrads = sparsecost*(repmat(hidmeans,numcases,1)-sparsetarget);
        
        penmeans = sparsedamping*penmeans + (1-sparsedamping)*posact{3}/numcases;
        sparsegrads2 = sparsecost*(repmat(penmeans,numcases,1)-sparsetarget2);
        
        labtopinc = momentum*labtopinc + ...
            epsilonw*( (poslabprods-negprodslabpen)/numcases - weightcost*dbn.rbm{2}.labtop);
        
        vishidinc = momentum*vishidinc + ...
            epsilonw*( (posprods{1}-negprods{1})/numcases - weightcost*dbn.rbm{1}.vishid - ...
            data'*sparsegrads/numcases );
        hidbiasinc = momentum*hidbiasinc + epsilonhb/numcases*(posact{2}-negact{2}) ...
            -epsilonhb/numcases*sum(sparsegrads);
        
        hidtopinc = momentum*hidtopinc + ...
            epsilonw*( (posprods{2}-negprods{2})/numcases - weightcost*dbn.rbm{2}.hidtop - ...
            posprobs{2}'*sparsegrads2/numcases - (posprobs{3}'*sparsegrads)'/numcases );
        penbiasinc = momentum*penbiasinc + epsilonhb/numcases*(posact{3}-negact{3}) ...
            -epsilonhb/numcases*sum(sparsegrads2);
        
        dbn.rbm{1}.vishid = dbn.rbm{1}.vishid + vishidinc;
        dbn.rbm{2}.hidtop = dbn.rbm{2}.hidtop + hidtopinc;
        dbn.rbm{2}.labtop = dbn.rbm{2}.labtop + labtopinc;
        dbn.rbm{1}.visbiases = dbn.rbm{1}.visbiases + visbiasinc;
        hidbiases = hidbiases + hidbiasinc;
        dbn.rbm{2}.topbiases = dbn.rbm{2}.topbiases + penbiasinc;
        dbn.rbm{2}.labbiases = dbn.rbm{2}.labbiases + labbiasinc;
        %%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
    if mod(epoch,10)==1 
        show_rbm(negstates{1}, numdims);
        title(['dbm_mf epoch ',num2str(epoch),' reconstruction ']);
        drawnow;
    end
    end

    fprintf(1, 'epoch %4i reconstruction error %6.1f \n Number of misclassified training cases %d (out of 60000) \n', epoch, errsum,60000-counter);
    errMF=[errMF;errsum];
    save  fullmnist_dbm
end;

end





