function rbm = randRBM( dimV, dimH, type )

if( strcmpi( 'GBRBM', type ) )
    rbm.type = 'GBRBM';
    rbm.vishid = randn(dimV, dimH) * 0.001;
    rbm.hidbiases = zeros(1, dimH);
    rbm.visbiases = zeros(1, dimV);
    rbm.z = ones(1, dimV);
else
    rbm.type = 'BBRBM';
    rbm.vishid = randn(dimV, dimH) * 0.001;
    rbm.hidbiases = zeros(1, dimH);
    rbm.visbiases = zeros(1, dimV);
end

end
