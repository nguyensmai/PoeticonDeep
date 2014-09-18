
function plotReconstruction(visdata)
numcases= size(visdata,1);
figure
n_basis_functions = 50;
time =2;
dt =0.005;
time_exec=2;
order =3;
load maxmind maxd mind
load temp_batch targets
for iSample=1:numcases
visible = reshape(visdata(iSample,:).*(maxd-mind)+mind,2,n_basis_functions);
subplot(9,ceil(numcases/9), iSample)
[ trajectory ] = dmpintegrate([ 0 1],[ 1 0],visible,time,dt,time_exec,order);
hold on
handle = plot(trajectory.y(:,1,1),trajectory.y(:,2,1));
title(find(targets(iSample,:)>0))
end

end