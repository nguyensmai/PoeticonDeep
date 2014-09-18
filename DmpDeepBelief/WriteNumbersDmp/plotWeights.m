%% weights for reconstruction
winv = pinv(w_class);
w3=[hidpen2'; hidgenbiases2]; 
w2=[hidpen'; hidgenbiases]; 
w1=[vishid'; visbiases];

%%

targets = eye(nTargets);
topprobs = targets*winv;



%%%%%%%%%%%%%%%%%%%% COMPUTE TRAINING RECONSTRUCTION
%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w3probs = 1./(1 + exp(-topprobs*w3)); w3probs = [w3probs  ones(nTargets,1)];
w2probs = 1./(1 + exp(-w3probs*w2)); 
w2states= w2probs > 0.5;
mu      = w2states*vishid' + repmat(visbiases, nTargets,1);
std = repmat(sqrt(exp(z)),nTargets,1);
dataout = random('norm', mu, std);

%%%% DISPLAY FIGURE RECONSTRUCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%
figure(3)
for iDigit=1:nTargets
subplot(3,3, iDigit)
theta = dataout(iDigit,:).*(maxd- mind) + mind;
[ trajectory ] = dmpintegrate([ 0 1],[ 1 0],theta,time,dt,time_exec,order);
hold on
handle = plot(trajectory.y(:,1,1),trajectory.y(:,2,1));
end