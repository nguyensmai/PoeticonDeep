function rbm = assocRBM(dimH, dimT, dimL)

    rbm.type = 'ASSOC';
    rbm.hidtop     = 0.1*randn(dimH, dimT);
    rbm.hidbiases  = zeros(1,dimH);
    
    rbm.labtop     = 0.1*randn(dimL,dimT);
    rbm.labbiases  = zeros(1,dimL);
    
    rbm.topbiases  = zeros(1,dimT);

end
