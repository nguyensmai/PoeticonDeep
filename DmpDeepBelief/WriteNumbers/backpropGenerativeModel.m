function [dbn, errL,negstates,neglabel]= backpropGenerativeModel(batchdata,batchtargets,dbn,maxepoch)

batch=1;
step =10^-3;
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

%untie
dbn.rbm{nLayers}.tophid = dbn.rbm{nLayers}.hidtop';
dbn.rbm{nLayers}.toplab = dbn.rbm{nLayers}.labtop';

for epoch=1:maxepoch
    errsum=0;
    targets=batchtargets(:,:,batch);
    for batch=1:numbatches
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
    
    %% 2. TOP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    i=nLayers;
    negstates{i+1} = posstates{i+1};
    
    
    %2.1. negative phase
    negprobs{i} = 1./(1 + exp(...
        -negstates{i+1}*dbn.rbm{i}.tophid ...
        -repmat(dbn.rbm{i}.hidbiases,numcases,1)...
        ));
    negstates{i} = negprobs{i}>rand(size(negprobs{i}));
    neglabel = exp( negstates{i+1}*dbn.rbm{i}.toplab...
        +repmat(dbn.rbm{i}.labbiases,numcases,1) );
    neglabel = neglabel./repmat(sum(neglabel,2),1,dbn.nTargets);
    
    %         %2.2. positive phase
    %         negprobs{i+1} = 1./(1 + exp( ...
    %             -negstates{i}*dbn.rbm{i}.hidtop...
    %             -neglabel*dbn.rbm{i}.labtop ...
    %             -repmat(dbn.rbm{i}.topbiases,numcases,1)...
    %             ));
    %         negstates{i+1} = negprobs{i+1} > rand(size(negprobs{i+1}));
    
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
    
    
    
    %% 5. update DBN with backprop
    % generative parameters : rbm 1
    delta1      = posstates{1} - negstates{1};
    iLayer= 1;
    wOprobs = [negstates{iLayer} ones(numcases,1)];   % numcases x (dimV +1) reformatting to compute with the weights and biases
    wIprobs = [negstates{iLayer+1} ones(numcases,1)]; % numcases x (dimH +1)
    w      = [dbn.rbm{iLayer}.hidvis ; dbn.rbm{iLayer}.visbiases];                  % dimH x dimV
    Ix  = delta1*w';
    delta0  = Ix;           % linear (gaussian) transfer function. numcases x (dimV +1)
    dw = (wIprobs'*delta1)/numcases;
    hidvisinc                 = dw(1:end-1,:);        % dimH x dimV
    visbiasinc                = dw(end,:);            % 1 x dimV
    dbn.rbm{iLayer}.hidvis    = dbn.rbm{iLayer}.hidvis + epsilonw*hidvisinc;
    dbn.rbm{iLayer}.visbiases = dbn.rbm{iLayer}.visbiases + epsilonvb*visbiasinc;
    delta1= delta0(:,1:end-1);
    
    % generative parameters
    for iLayer= 2:nLayers-1
        wOprobs = [negstates{iLayer} ones(numcases,1)];   % numcases x (dimV +1) reformatting to compute with the weights and biases
        wIprobs = [negstates{iLayer+1} ones(numcases,1)]; % numcases x (dimH +1)
        w      = [dbn.rbm{iLayer}.hidvis ; dbn.rbm{iLayer}.visbiases];                  % dimH x dimV
        Ix  = delta1*w';
        delta0  = Ix.*wIprobs.*(1-wIprobs);           % sigmoid transfer function. numcases x (dimV +1)
        dw = (wIprobs'*delta1)/numcases;
        hidvisinc                 = dw(1:end-1,:);        % dimH x dimV
        visbiasinc                = dw(end,:);            % 1 x dimV
        dbn.rbm{iLayer}.hidvis    = dbn.rbm{iLayer}.hidvis + epsilonw*hidvisinc;
        dbn.rbm{iLayer}.visbiases = dbn.rbm{iLayer}.visbiases + epsilonvb*visbiasinc;
        delta1= delta0(:,1:end-1);
    end
    
    %top layer: hidden
    iLayer=nLayers;
    wOprobs = [negstates{iLayer} ones(numcases,1)];   % numcases x (dimV +1) reformatting to compute with the weights and biases
    wIprobs = [negstates{iLayer+1} ones(numcases,1)]; % numcases x (dimH +1)
    w      = [dbn.rbm{iLayer}.tophid ; dbn.rbm{iLayer}.hidbiases];                  % dimH x dimV
    Ix  = delta1*w';
    delta0  = Ix.*wIprobs.*(1-wIprobs);           % sigmoid transfer function. numcases x (dimV +1)
    dw = (wIprobs'*delta1)/numcases;
    tophidinc                  = dw(1:end-1,:);        % dimH x dimV
    hidbiasinc                 = dw(end,:);            % 1 x dimV
    dbn.rbm{nLayers}.tophid    = dbn.rbm{nLayers}.tophid + epsilonw*tophidinc;
    dbn.rbm{nLayers}.hidbiases = dbn.rbm{nLayers}.hidbiases + epsilonvb*hidbiasinc;
    deltaH  = delta0(:,1:end-1);
    
    %top layer : labels
    delta1     = targets - neglabel;
    wOprobs = [negstates{iLayer} ones(numcases,1)];   % numcases x (dimV +1) reformatting to compute with the weights and biases
    wIprobs = [negstates{iLayer+1} ones(numcases,1)]; % numcases x (dimH +1)
    w      = [dbn.rbm{iLayer}.toplab ; dbn.rbm{iLayer}.labbiases];                  % dimH x dimV
    Ix  = delta1*w';
    delta0  = Ix.*wIprobs.*(1-wIprobs);           % sigmoid transfer function. numcases x (dimV +1)
    dw = (wIprobs'*delta1)/numcases;
    toplabinc                  = dw(1:end-1,:);        % dimH x dimV
    labbiasinc                 = dw(end,:);            % 1 x dimV
    dbn.rbm{nLayers}.toplab    = dbn.rbm{nLayers}.toplab    + epsilonwlt*toplabinc;
    dbn.rbm{nLayers}.labbiases = dbn.rbm{nLayers}.labbiases + epsilontlb*labbiasinc;
    deltaL= delta0(:,1:end-1);
    
    
    
    %top layer: top
    delta1 = deltaL+deltaH;
    wOprobs = [negstates{iLayer+1} ones(numcases,1)];   % numcases x (dimV +1) reformatting to compute with the weights and biases
    wIprobs = [negstates{iLayer} ones(numcases,1)]; % numcases x (dimH +1)
    w      = [dbn.rbm{iLayer}.hidtop ; dbn.rbm{iLayer}.topbiases];                  % dimH x dimV
    Ix  = delta1*w';
    delta0  = Ix.*wIprobs.*(1-wIprobs);           % sigmoid transfer function. numcases x (dimV +1)
    dw = (wIprobs'*delta1)/numcases;
    hidtopinc                  = dw(1:end-1,:);        % dimH x dimV
    topbiasinc                 = dw(end,:);            % 1 x dimV
    dbn.rbm{nLayers}.hidtop    = dbn.rbm{nLayers}.hidtop + epsilonw*hidtopinc;
    dbn.rbm{nLayers}.topbiases = dbn.rbm{nLayers}.topbiases + epsilonvb*topbiasinc;
    delta1= delta0(:,1:end-1);
    
    
    % recognition parameters
    for iLayer=nLayers-1:-1:1
        wOprobs = [negstates{iLayer+1} ones(numcases,1)];   % numcases x (dimV +1) reformatting to compute with the weights and biases
        wIprobs = [negstates{iLayer} ones(numcases,1)]; % numcases x (dimH +1)
        w      = [dbn.rbm{iLayer}.vishid ; dbn.rbm{iLayer}.hidbiases];                  % dimH x dimV
        Ix  = delta1*w';
        delta0  = Ix.*wIprobs.*(1-wIprobs);           % sigmoid transfer function. numcases x (dimV +1)
        dw = (wIprobs'*delta1)/numcases;
        vishidinc                  = dw(1:end-1,:);        % dimH x dimV
        hidbiasinc                 = dw(end,:);            % 1 x dimV
        dbn.rbm{iLayer}.vishid    = dbn.rbm{iLayer}.vishid + epsilonw*vishidinc;
        dbn.rbm{iLayer}.hidbiases = dbn.rbm{iLayer}.hidbiases + epsilonvb*hidbiasinc;
        delta1= delta0(:,1:end-1);
    end
    
    %% 6. error compute
    err1= sum(sum( (targets - neglabel).^2 ));
    err2= sum(sum( (posstates{1}-negstates{1}).^2 ));
    errsum= errsum+err1+err2;
    end
    
    errL=[errL;errsum,err1,err2];
    
    if mod(epoch,10)==0
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