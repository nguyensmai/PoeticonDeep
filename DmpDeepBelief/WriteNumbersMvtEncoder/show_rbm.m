% Copyright (c) 2013 Sao Mai Nguyen
%               e-mail : nguyensmai@gmail.com
%               http://nguyensmai.free.fr/%%
function show_rbm(data,numdims)
datasize=size(data,1);
nTargets=10;
load maxmin

t=[0:500]';


for iSample=1:datasize
    
    visible= data(iSample,:);
    visible = visible.*[(maxdx-mindx)*ones(1,numdims/2) (maxdy-mindy)*ones(1,numdims/2)] ...
        +[mindx*ones(1,numdims/2) mindy*ones(1,numdims/2)];
    a1 = visible(:,1:numdims/2);
    a2 = visible(:,numdims/2+1:end);
    subplot(ceil(datasize/nTargets), nTargets,iSample)
    Vinv1 = MovementEncoder.getMovementTrajectory(a1,t);
    Vinv2 = MovementEncoder.getMovementTrajectory(a2,t);
    handle= plot(Vinv1,Vinv2, '-b');
end

end