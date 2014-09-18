% Copyright (c) 2013 Sao Mai Nguyen
%               e-mail : nguyensmai@gmail.com
%               http://nguyensmai.free.fr/

function dbn= randDBN(nodes,nTargets, type)
dbn.rbm      = cell( numel(nodes)-1, 1 );
dbn.nodes    = nodes;
dbn.type     = type;
dbn.nTargets = nTargets;

i = 1;
if( strcmpi( 'GBRBM', type ) )
 dbn.rbm{i} = randRBM( nodes(i), nodes(i+1), 'GBRBM' );
else
 dbn.rbm{i} = randRBM( nodes(i), nodes(i+1), 'BBRBM' );
end

for i=2:numel(dbn.rbm) - 1
 dbn.rbm{i} = randRBM( nodes(i), nodes(i+1), 'BBRBM' );
end

i = numel(dbn.rbm);
dbn.rbm{i} = assocRBM( nodes(i), nodes(i+1),nTargets );


end
