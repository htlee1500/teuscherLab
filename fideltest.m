Ncells=28*28;
dummyv=ones(Ncells,1);
%generates a connectivity matrix of nearest neighbors with a grid size of 8
%cells square
%conM=full(spdiags([dummyv dummyv dummyv dummyv],[-28 -1 1 28 ],Ncells,Ncells));
%toroidal boundary conditions
conM=full(spdiags(repmat(dummyv,1,8),[-(Ncells-28) -28 -27 -1 1 27 28 Ncells-28],Ncells,Ncells));
%all
%conM=ones(Ncells,Ncells).*~eye(Ncells,Ncells);
rM=(2*rand(Ncells)-1);
SimProp.iWeights=1*conM.*rM;%0.2*ones(Ncells,Ncells).*~eye(Ncells,Ncells);
SimProp.Namp = 0.0;%0.03;%*SimProp.gam; %sig value 0.001
SimProp.v0=8.5+2*(2*rand(1,Ncells)-1);
SimProp.vrest=-0;
SimProp.Algorithm='GL';
SimProp.dt = 0.0001;
NetProp.Ncells=Ncells;
NetProp.vTh=10;%mV
NetProp.vPeak=40;
NetProp.Rm=40;
NetProp.TauM=10;%50
NetProp.iRefrac=0.5;5;%10; %ms
biasI=[1.5 ];
ampV= [0.3 ];
SimProp.alpha=0.1;
preT=[100 ];
stimIn=ampV.*ones(1, Ncells)+biasI;
preTM=biasI.*ones(preT/SimProp.dt,Ncells);
%stimInL=repmat([stimIn'],1,3);
SimProp.IinjDC= [preTM' stimIn'];
tstim=[0:SimProp.dt:(length(SimProp.IinjDC)-SimProp.dt)];
SimProp.tEnd=size(SimProp.IinjDC,2)*SimProp.dt;
SimProp.t=(SimProp.tStart:SimProp.dt:SimProp.tEnd-SimProp.dt)';
SimProp.WinSize=length(SimProp.t);
out(b)=FractionalIntegrateandFire(NetProp,SimProp); % main output
