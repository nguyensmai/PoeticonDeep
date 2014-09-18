function show_rbm(batchdata,numdims)
totnum=size(batchdata,1);
load maxmin
order = 3;
time =2;
dt=0.02;
time_exec=time;
% n_basis_functions=5;
% nRepeat=1;

for iSample=1:totnum
    coordsX1=mean(batchdata(iSample,1:(2*n_basis_functions+4):end));
    coordsY1=mean(batchdata(iSample,2:(2*n_basis_functions+4):end));
    coordsX2=mean(batchdata(iSample,3:(2*n_basis_functions+4):end));
    coordsY2=mean(batchdata(iSample,4:(2*n_basis_functions+4):end));
        
    visible= batchdata(iSample,:);
    visible= reshape(visible,2*n_basis_functions+4,nRepeat)';
    visible= visible(:,5:end);
     visible = visible + 0.005*(rand(size(visible))-0.5); %testing stability

    visible= mean(visible,1);
    visible = visible.*[(maxdx-mindx)*ones(1,n_basis_functions) (maxdy-mindy)*ones(1,n_basis_functions)] ...
        +[mindx*ones(1,n_basis_functions) mindy*ones(1,n_basis_functions)];
    visible = reshape(visible,2,n_basis_functions);
    subplot(9,ceil(totnum/9), iSample)
    [ trajectory ] = dmpintegrate([ coordsX1 coordsY1],...
        [coordsX2 coordsY2 ],visible,time,dt,time_exec,order);
    hold on
    handle = plot(trajectory.y(:,1,1),trajectory.y(:,2,1),'r');
end

end