function show_rbm(batchdata,numdims, batchtargets)
totnum=size(batchdata,1);
load maxmin
order = 3;
time =2;
dt=0.02;
time_exec=time;
% n_basis_functions=5;
% nRepeat=1;
errsum=0;

for iSample=1:totnum
    coordsX1=mean(batchdata(iSample,1:(2*n_basis_functions+4):end));
    coordsY1=mean(batchdata(iSample,2:(2*n_basis_functions+4):end));
    coordsX2=mean(batchdata(iSample,3:(2*n_basis_functions+4):end));
    coordsY2=mean(batchdata(iSample,4:(2*n_basis_functions+4):end));
        
    visible= batchdata(iSample,:);
    visible= reshape(visible,2*n_basis_functions+4,nRepeat)';
    visible= visible(:,5:end);
%     visibleN = visible + 0.05*(rand(size(visible))-0.5); %testing stability
%     err = sum(sum( (visible-visibleN).^2 ))/(numdims);
%     errsum = errsum + err;           
%     visible=visibleN;
    visible= mean(visible,1);
    visible = visible.*[(maxdx-mindx) (maxdy-mindy)] ...
        +[mindx mindy];
    visible = reshape(visible,2,n_basis_functions);
    subplot(9,ceil(totnum/9), iSample)
    [ trajectory ] = dmpintegrate([ coordsX1 coordsY1],...
        [coordsX2 coordsY2 ],visible,time,dt,time_exec,order);
    hold on
    handle = plot(trajectory.y(:,1,1),trajectory.y(:,2,1),'r');
    if nargin>2 && ~isempty(batchtargets)    
        title(find(batchtargets(iSample,:,1)>0))
    end

end
% errsum/totnum
end