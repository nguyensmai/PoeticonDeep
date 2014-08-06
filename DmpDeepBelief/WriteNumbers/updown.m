function [dbn, err, err1,err2]= updown(batchdata,targets,dbn)

batch=1;
numCDiters=2;%100;
epsilonw      = 0.01;   % Learning rate for weights
epsilonvb     = 0.01;   % Learning rate for biases of visible units
epsilonhb     = 0.01;   % Learning rate for biases of hidden units
weightcost  = 0; %0.2;
[numcases numdims numbatches]=size(batchdata);
nLayers   = numel(dbn.rbm);
posprobs  = cell(nLayers+1,1);
negprobs  = cell(nLayers+1,1);
posstates = cell(nLayers+1,1);
negstates = cell(nLayers+1,1);
pnegprobs = cell(nLayers+1,1);
pposprobs = cell(nLayers+1,1);

%% 1. bottom to top
posstates{1}  = batchdata(:,:,batch);

for i=1:nLayers - 1
    posprobs{i+1} = 1./(1 + exp( ...
        -posstates{i}*dbn.rbm{i}.vishid - repmat(dbn.rbm{i}.hidbiases,numcases,1)...
        ));
    posstates{i+1} = posprobs{i+1} > rand(size(posprobs{i+1}));
end

% top
i=nLayers;
posprobs{i+1} = 1./(1 + exp( ...
    -posstates{i}*dbn.rbm{i}.hidtop ...
    -targets*dbn.rbm{i}.labtop ...
    -repmat(dbn.rbm{i}.topbiases,numcases,1)...
    ));
posstates{i+1} = posprobs{i+1} > rand(size(posprobs{i+1}));

posprods = double(posstates{i})' * double(posstates{i+1});
poslabprods = double(targets)' * double(posstates{i+1});

%% 2. top: gibbs sampling
i=nLayers;
negstates{i+1} = posstates{end};

for iter =1:numCDiters
    
    %negative phase
    negprobs{i} = 1./(1 + exp(...
        -negstates{i+1}*dbn.rbm{i}.hidtop' ...
        -repmat(dbn.rbm{i}.hidbiases,numcases,1)...
        ));
    negstates{i} = negprobs{i}>rand(size(negprobs{i}));
    neglabel = exp( negstates{i+1}*dbn.rbm{i}.labtop'...
        +repmat(dbn.rbm{i}.labbiases,numcases,1) );
    neglabel = neglabel./repmat(sum(neglabel,2),1,dbn.nTargets);
    
    %positive phase
    negprobs{i+1} = 1./(1 + exp( ...
        -negstates{i}*dbn.rbm{i}.hidtop...
        -targets*dbn.rbm{i}.labtop ...
        -repmat(dbn.rbm{i}.topbiases,numcases,1)...
        ));
    negstates{i+1} = negprobs{i+1} > rand(size(negprobs{i+1}));

end
negprods    = double(negstates{i})'*double(negstates{i+1});
neglabprods = double(neglabel)'*double(negstates{i+1});


%% 3. top to bottom
for i=(nLayers-1):-1:2
    negprobs{i} = 1./(1 + exp(...
        -negstates{i+1}*dbn.rbm{i}.vishid' ...
        -repmat(dbn.rbm{i}.visbiases,numcases,1)...
        ));
    negstates{i} = negprobs{i} > rand(size(negprobs{i}));
end

% gaussian bottom layer
i=1;
if( strcmpi( 'GBRBM', dbn.rbm{i}.type ) )
    negmu    = negstates{i+1}*dbn.rbm{i}.vishid' + repmat(dbn.rbm{i}.visbiases,numcases,1);
    std = repmat(sqrt(exp(dbn.rbm{i}.z)), numcases,1);
    negprobs{i}  = negmu; %random('norm', negmu, std);
else
    negprobs{i} = 1./(1 + exp(...
        -negstates{i+1}*dbn.rbm{i}.vishid' ...
        -repmat(dbn.rbm{i}.visbiases,numcases,1)...
        ));
end
negstates{i} = negprobs{i} > rand(size(negprobs{i}));


%% 4. predictions
for i=1:nLayers-1
pnegprobs{i+1} = 1./(1+exp(...
   - negstates{i}*dbn.rbm{i}.vishid - repmat(dbn.rbm{i}.hidbiases,numcases,1)...
   ));

pposprobs{i}= 1./(1+exp(...
   - posstates{i+1}*dbn.rbm{i}.vishid'...
   - repmat(dbn.rbm{i}.visbiases,numcases,1)...
   ));
end

%% 5. update DBN
%updates to prediction parameters
i=1;
if( strcmpi( 'GBRBM', dbn.rbm{1}.type ) )
z = dbn.rbm{1}.z;
data = posstates{1};
vishid = dbn.rbm{1}.vishid;
vishidinc  = negprobs{i}'*(negstates{i+1}-pnegprobs{i+1})./(repmat(exp(z)',1,dbn.nodes(2))*numcases) - weightcost*vishid;
hidbiasinc = sum(negstates{i+1}-pnegprobs{i+1})/numcases;
dbn.rbm{i}.vishid = vishid + epsilonw/100*vishidinc;
dbn.rbm{i}.hidbiases = dbn.rbm{i}.hidbiases + epsilonhb/100*hidbiasinc;
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
i=1;
if( strcmpi( 'GBRBM', dbn.rbm{1}.type ) )
z = dbn.rbm{1}.z;
data =posstates{1};
vishid =dbn.rbm{1}.vishid;
hidvisinc  = posstates{i+1}'*(posstates{i}-pposprobs{i})./(repmat(exp(z),dbn.nodes(2),1)*numcases) - weightcost*vishid';
visbiasinc = sum(posstates{i}-pposprobs{i})./(exp(z)*numcases);
zinc = exp(-z) .*(...
    mean(1/2*(data-repmat(visbiasinc,numcases,1)).^2 - data.*(posstates{i+1}*vishid'),1) ...
    - mean(1/2*(pposprobs{1}-repmat(visbiasinc,numcases,1)).^2 - pposprobs{1}.*(pposprobs{2}*vishid'),1) ...
    );
dbn.rbm{1}.z = dbn.rbm{1}.z+zinc;
dbn.rbm{i}.vishid = dbn.rbm{i}.vishid + epsilonw/100*hidvisinc';
dbn.rbm{i}.visbiases = dbn.rbm{i}.visbiases + epsilonvb/100*visbiasinc;    
else
hidvisinc  = posstates{i+1}'*(posstates{i}-pposprobs{i})/numcases - weightcost*(dbn.rbm{i}.vishid)';
visbiasinc = (posstates{i}-pposprobs{i})/numcases;
dbn.rbm{i}.vishid = dbn.rbm{i}.vishid + epsilonw*hidvisinc';
dbn.rbm{i}.visbiases = dbn.rbm{i}.visbiases + epsilonvb*visbiasinc;
end

for i=2:nLayers-1
hidvisinc  = posstates{i+1}'*(posstates{i}-pposprobs{i})/numcases - weightcost*(dbn.rbm{i}.vishid)';
visbiasinc = sum(posstates{i}-pposprobs{i})/numcases;
dbn.rbm{i}.vishid = dbn.rbm{i}.vishid + epsilonw*hidvisinc';
dbn.rbm{i}.visbiases = dbn.rbm{i}.visbiases + epsilonvb*visbiasinc;
end

%updates on the top layer
dbn.rbm{nLayers}.labtop = dbn.rbm{nLayers}.labtop + epsilonw*(poslabprods-neglabprods - weightcost*dbn.rbm{nLayers}.labtop);
dbn.rbm{nLayers}.hidtop = dbn.rbm{nLayers}.hidtop + epsilonw*(posprods-negprods - weightcost*dbn.rbm{nLayers}.hidtop);
dbn.rbm{nLayers}.topbiases = dbn.rbm{nLayers}.topbiases + epsilonhb*sum(posstates{nLayers+1}-negstates{nLayers+1});
dbn.rbm{nLayers}.labbiases = dbn.rbm{nLayers}.labbiases + epsilonvb*sum(targets - neglabel);
dbn.rbm{nLayers}.hidbiases = dbn.rbm{nLayers}.hidbiases + epsilonvb*sum(posstates{nLayers}-negstates{nLayers});

%% error compute
err1= sum(sum( (targets - neglabel).^2 ));
err2= sum(sum( (posstates{1}-negstates{1}).^2 ));
err= err1+err2;

end