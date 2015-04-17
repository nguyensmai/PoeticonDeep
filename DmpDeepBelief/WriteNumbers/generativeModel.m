function visible = generativeModel(dbn,label)
targets=zeros(1,dbn.nTargets);
targets(label)= 1;

numCDiters=100;
numcases =1;

nLayers   = numel(dbn.rbm);
posprobs  = cell(nLayers+1,1);
negprobs  = cell(nLayers+1,1);
posstates = cell(nLayers+1,1);
negstates = cell(nLayers+1,1);
pnegprobs = cell(nLayers+1,1);
pposprobs = cell(nLayers+1,1);


%1. top
i=nLayers;
posstates{i} = double(rand(size(dbn.rbm{i}.hidbiases))>0.5);
posprobs{i+1} = 1./(1 + exp( ...
    -posstates{i}*dbn.rbm{i}.hidtop ...
    -targets*dbn.rbm{i}.labtop ...
    -repmat(dbn.rbm{i}.topbiases,numcases,1)...
    ));
posstates{i+1} = posprobs{i+1} > rand(size(posprobs{i+1}));


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
    negprobs{i}  = random('norm', negmu, std);
else
    negprobs{i} = 1./(1 + exp(...
        -negstates{i+1}*dbn.rbm{i}.vishid' ...
        -repmat(dbn.rbm{i}.visbiases,numcases,1)...
        ));
end
negstates{i} = negprobs{i} > rand(size(negprobs{i}));

visible= negstates{1};

end
