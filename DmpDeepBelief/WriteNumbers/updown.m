function [dbn, errL]= updown(batchdata,targets,dbn,maxepoch)

batch=1;
numCDiters=10;
step =10^-4;
epsilonw      = step;      % Learning rate for weights
epsilonvb     = step;      % Learning rate for biases of visible units
epsilonhb     = step;      % Learning rate for biases of hidden units
epsilonz     = 10^-7;       % Learning rate for biases of hidden units
weightcost  = 0.02;
epsilontlb     = 5*step;   % Learning rate for biases of label units for top rbm
epsilonttb     = 5*step;   % Learning rate for biases of top units for top rbm
epsilonthb     = step;     % Learning rate for biases of hidden units for top rbm
epsilonwlt     = 5*step;   % Learning rate for weights between labels and top
epsilonwht     = step;     % Learning rate for weights between hidden and top
initialmomentum  = 0.3;
finalmomentum    = 0.9;
[numcases numdims numbatches]=size(batchdata);
nLayers   = numel(dbn.rbm);
posprobs  = cell(nLayers+1,1);
negprobs  = cell(nLayers+1,1);
posstates = cell(nLayers+1,1);
negstates = cell(nLayers+1,1);
pnegprobs = cell(nLayers+1,1);
pposprobs = cell(nLayers+1,1);
laptopinc = 0;
hidtopinc = 0;
topbiasinc = 0;
labbiasinc = 0;
hidbiasinc = 0;
errL =[];

figure('name','updown')

for epoch=1:maxepoch
    %% 1. BOTTOM-UP PASS TO GET WAKE/POSITIVE PHASE PROB & SAMPLE STATES %%%%%
    posstates{1}  = batchdata(:,:,batch);
    
    %1.1 First Bernoulli RBM
    if( strcmpi( 'GBRBM', dbn.rbm{1}.type ) )
        posstates{1}= posstates{1}./repmat(exp(dbn.rbm{1}.z),numcases,1);
    end
    
    %1.2. Sigmoid Belief Networks
    for i=1:nLayers - 1
        posprobs{i+1} = 1./(1 + exp( ...
            -posstates{i}*dbn.rbm{i}.vishid...
            - repmat(dbn.rbm{i}.hidbiases,numcases,1)...
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
    
    posprods = double(posstates{i})' * double(posstates{i+1});
    poslabprods = double(targets)' * double(posstates{i+1});
    
    %% 2. TOP: GIBBS SAMPLING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    i=nLayers;
    negstates{i+1} = posstates{i+1};
    
    for iter =1:numCDiters
        
        %2.1. negative phase
        negprobs{i} = 1./(1 + exp(...
            -negstates{i+1}*dbn.rbm{i}.hidtop' ...
            -repmat(dbn.rbm{i}.hidbiases,numcases,1)...
            ));
        negstates{i} = negprobs{i}>rand(size(negprobs{i}));
        neglabel = exp( negstates{i+1}*dbn.rbm{i}.labtop'...
            +repmat(dbn.rbm{i}.labbiases,numcases,1) );
        neglabel = neglabel./repmat(sum(neglabel,2),1,dbn.nTargets);
        
        %2.2. positive phase
        negprobs{i+1} = 1./(1 + exp( ...
            -negstates{i}*dbn.rbm{i}.hidtop...
            -neglabel*dbn.rbm{i}.labtop ...
            -repmat(dbn.rbm{i}.topbiases,numcases,1)...
            ));
        negstates{i+1} = negprobs{i+1} > rand(size(negprobs{i+1}));
        
    end
    negprods    = double(negstates{i})'*double(negstates{i+1});
    neglabprods = double(neglabel)'*double(negstates{i+1});
    
    
    %% 3. top to bottom
    for i=(nLayers-1):-1:2
        negprobs{i} = 1./(1 + exp(...
            -negstates{i+1}*dbn.rbm{i}.hidvis ...
            -repmat(dbn.rbm{i}.visbiases,numcases,1)...
            ));
        negstates{i} = negprobs{i} > rand(size(negprobs{i}));
    end
    
    % gaussian bottom layer
    i=1;
    if( strcmpi( 'GBRBM', dbn.rbm{i}.type ) )
        negmu    = negstates{i+1}*dbn.rbm{i}.hidvis + repmat(dbn.rbm{i}.visbiases,numcases,1);
        %std = repmat(sqrt(exp(dbn.rbm{i}.z)), numcases,1);
        negprobs{i}  = negmu; %random('norm', negmu, std);
        negstates{i} = negprobs{i} ;
    else
        negprobs{i} = 1./(1 + exp(...
            -negstates{i+1}*dbn.rbm{i}.hidvis ...
            -repmat(dbn.rbm{i}.visbiases,numcases,1)...
            ));
        negstates{i} = negprobs{i} > rand(size(negprobs{i}));
    end
    
    
    %% 4. predictions
    %1.1 First Bernoulli RBM
    if( strcmpi( 'GBRBM', dbn.rbm{i}.type ) )
        negstates{1}= negstates{1}./repmat(exp(dbn.rbm{1}.z),numcases,1);
    end
    
    for i=1:nLayers-1
        pnegprobs{i+1} = 1./(1+exp(...
            - negstates{i}*dbn.rbm{i}.vishid ...
            - repmat(dbn.rbm{i}.hidbiases,numcases,1)...
            ));
        
        pposprobs{i}= 1./(1+exp(...
            - posstates{i+1}*dbn.rbm{i}.hidvis...
            - repmat(dbn.rbm{i}.visbiases,numcases,1)...
            ));
    end
    
    i=1;
    if( strcmpi( 'GBRBM', dbn.rbm{i}.type ) )
        negmu    = posstates{i+1}*dbn.rbm{i}.hidvis + repmat(dbn.rbm{i}.visbiases,numcases,1);
        %std = repmat(sqrt(exp(dbn.rbm{i}.z)), numcases,1);
        pposprobs{i}  = negmu; %random('norm', negmu, std);
    else
        pposprobs{i} = 1./(1 + exp(...
            -posstates{i+1}*dbn.rbm{i}.hidvis ...
            -repmat(dbn.rbm{i}.visbiases,numcases,1)...
            ));
    end
    
    %% 5. update DBN
    if epoch>10,
        momentum=finalmomentum;
    else%if epoch>20
        momentum=initialmomentum;
    end;
    
    
    %updates to recognition parameters
    i=1;
    if( strcmpi( 'GBRBM', dbn.rbm{1}.type ) )
        z = dbn.rbm{1}.z;
        data = posstates{1};
        vishid = dbn.rbm{1}.vishid;
        vishidinc  = negprobs{i}'*(negstates{i+1}-pnegprobs{i+1})./(numcases) - weightcost*vishid;
        hidbiasinc = sum(negstates{i+1}-pnegprobs{i+1})/numcases;
        dbn.rbm{i}.vishid = vishid + epsilonw/10*vishidinc;
        dbn.rbm{i}.hidbiases = dbn.rbm{i}.hidbiases + epsilonhb/10*hidbiasinc;
    else
        vishidinc  = negprobs{i}'*(negstates{i+1}-pnegprobs{i+1})/numcases - weightcost*vishid;
        hidbiasinc = sum(negstates{i+1}-pnegprobs{i+1})/numcases;
        dbn.rbm{i}.vishid = dbn.rbm{i}.vishid + epsilonw*vishidinc;
        dbn.rbm{i}.hidbiases = dbn.rbm{i}.hidbiases + epsilonhb*hidbiasinc;
    end
    
    
    for i=2:nLayers-1
        vishidinc  = negprobs{i}'*(negstates{i+1}-pnegprobs{i+1})/numcases - weightcost*dbn.rbm{i}.vishid;
        hidbiasinc = sum(negstates{i+1}-pnegprobs{i+1})/numcases;
        dbn.rbm{i}.vishid = dbn.rbm{i}.vishid + epsilonw*vishidinc;
        dbn.rbm{i}.hidbiases = dbn.rbm{i}.hidbiases + epsilonhb*hidbiasinc;
    end
    
    %updates to generative parameters
    %     i=1;
    %     if( strcmpi( 'GBRBM', dbn.rbm{1}.type ) )
    %         z          = dbn.rbm{1}.z;
    %         data       = posstates{1};
    %         hidvis     = dbn.rbm{1}.hidvis;
    %         visbiases  = dbn.rbm{1}.visbiases;
    %         hidvisinc  = posstates{i+1}'*(posstates{i}-pposprobs{i})./(numcases) - weightcost*hidvis;
    %         visbiasinc = sum(posstates{i}-pposprobs{i})./(exp(z)*numcases);
    %         zinc = exp(-z) .*(...
    %             mean(1/2*(data-repmat(visbiases,numcases,1)).^2 - data.*(posstates{i+1}*hidvis),1) ...
    %             - mean(1/2*(pposprobs{1}-repmat(visbiases,numcases,1)).^2 - pposprobs{1}.*(pposprobs{2}*hidvis),1) ...
    %             );
    %         dbn.rbm{1}.z = dbn.rbm{1}.z+epsilonz*zinc;
    %         dbn.rbm{i}.hidvis = dbn.rbm{i}.hidvis + epsilonw/10*hidvisinc;
    %         dbn.rbm{i}.visbiases = dbn.rbm{i}.visbiases + epsilonvb/10*visbiasinc;
    %     else
    %         hidvisinc  = posstates{i+1}'*(posstates{i}-pposprobs{i})/numcases - weightcost*(dbn.rbm{i}.hidvis);
    %         visbiasinc = (posstates{i}-pposprobs{i})/numcases;
    %         dbn.rbm{i}.hidvis = dbn.rbm{i}.hidvis + epsilonw*hidvisinc;
    %         dbn.rbm{i}.visbiases = dbn.rbm{i}.visbiases + epsilonvb*visbiasinc;
    %     end
    
    for i=2:nLayers-1
        hidvisinc  = posstates{i+1}'*(posstates{i}-pposprobs{i})/numcases - weightcost*(dbn.rbm{i}.hidvis);
        visbiasinc = sum(posstates{i}-pposprobs{i})/numcases;
        dbn.rbm{i}.hidvis = dbn.rbm{i}.hidvis + epsilonw*hidvisinc;
        dbn.rbm{i}.visbiases = dbn.rbm{i}.visbiases + epsilonvb*visbiasinc;
    end
    
    
    
    %updates on the top layer
    laptopinc = momentum*laptopinc + (poslabprods-neglabprods - weightcost*dbn.rbm{nLayers}.labtop);
    hidtopinc = momentum*hidtopinc + (posprods-negprods - weightcost*dbn.rbm{nLayers}.hidtop);
    topbiasinc = momentum*topbiasinc + sum(posstates{nLayers+1}-negstates{nLayers+1});
    labbiasinc = momentum*labbiasinc + sum(targets - neglabel);
    hidbiasinc = momentum*hidbiasinc + sum(posstates{nLayers}-negstates{nLayers});
    dbn.rbm{nLayers}.labtop    = dbn.rbm{nLayers}.labtop    + epsilonwlt*laptopinc;
    dbn.rbm{nLayers}.hidtop    = dbn.rbm{nLayers}.hidtop    + epsilonwht*hidtopinc;
    dbn.rbm{nLayers}.topbiases = dbn.rbm{nLayers}.topbiases + epsilonttb*topbiasinc;
    dbn.rbm{nLayers}.labbiases = dbn.rbm{nLayers}.labbiases + epsilontlb*labbiasinc;
    dbn.rbm{nLayers}.hidbiases = dbn.rbm{nLayers}.hidbiases + epsilonthb*hidbiasinc;
    
    %% error compute
    err1= sum(sum( (targets - neglabel).^2 ));
    err2= sum(sum( (posstates{1}-negstates{1}).^2 ));
    errsum= err1+err2;
    errL=[errL;errsum,err1,err2];
    
    if mod(epoch,100)==0
        fprintf(1, 'UPDOWN: epoch %4i error %6.1f  \n', epoch, errsum);
        plot(epoch, errsum,'xb');
        hold on;
        plot(epoch, err1,'xg');
        plot(epoch, err2,'xr');
        drawnow
    end
    
end
%plotReconstruction(negstates{1});

end