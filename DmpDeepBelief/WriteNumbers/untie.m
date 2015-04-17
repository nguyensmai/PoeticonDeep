function dbn = untie(dbn)

nLayers   = numel(dbn.rbm);

for iLayer=1:nLayers-1
    dbn.rbm{iLayer}.hidvis = dbn.rbm{iLayer}.vishid';
end



end
