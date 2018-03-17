test=load("test.txt");
train=load("train.txt");
inp_data=train(:,1);
tar_data=train(:,2);
X=zeros(10,20);
for i =1:10
  for t =1:20
    X(i,t)=inp_data(t)^(i-1);
endfor
endfor
xStart = 0;
dx1 =0.0003/6;
N_iter = 20000;
lambda = xStart + (0:N_iter-1)*dx1;

tr=train(1:15,:);
val=train(16:20,:);

tr_targ=tr(:,2);
tr_inp=tr(:,1);
val_tar=val(:,2);

cost_simple=zeros(1,length(lambda));
X1=X(:,1:15);
Xout=X(:,16:20);

for u =1:length(lambda)
  
  w_star=inv(X1*X1'+lambda(u)*eye(10,10))*X1*tr_targ;
  %cost_simple(:,u) = (2*(norm( Xout'*w_star-val_tar)^2 +lambda(u)*norm( w_star )^2 ) /5)^0.5; %bias included
  cost_simple(:,u) = (2*(norm( Xout'*w_star-val_tar) +lambda(u)*norm( w_star(2:10) ) ) /5)^0.5;
endfor
[M1,I1]=min(cost_simple);
la=(I1-1)*dx1;
subplot(2,3,1);
i=1:N_iter;
plot(log((i-1)*dx1),cost_simple(i));

ylabel('RMS cost');
xlabel('log(lambda)');
title('Simple Split');

te_inp=test(:,1);
te_targ=test(:,2);

X_te=zeros(10,size(test)(1));
for i =1:10
  for t =1:size(test)(1)
    X_te(i,t)=te_inp(t)^(i-1);
endfor
endfor
w_st=inv((X*X'+la*eye(10,10)))*X*tar_data;
%cost_test_simple=norm(X_te'*w_st-te_targ)^2+la*norm(w_st)^2;
cost_test_simple=norm(X_te'*w_st-te_targ)^2+la*norm(w_st(2:10))^2;

%K-fold split

xStart = 1;
dx = 1;
N = 20;
f = xStart + (0:N-1)*dx;
k_split=f(randperm(length(f)));
X=zeros(10,20);
for i =1:10
  for t =1:20
    X(i,t)=inp_data(k_split(t))^(i-1);
k_split_cost=zeros(N_iter,1);
ktar_data=zeros(20,1);
endfor
endfor
for y =1:20
  ktar_data(y)=tar_data(k_split(y));
endfor
for u =1:N_iter
  cost1=0;
  for m = 1:10
    
    U1=X(:,1:2*m-2);
    U2=X(:,2*m+1:20);
    U=[U1 U2];
    ksplit_tr_targ=[ktar_data(1:2*m-2);ktar_data(2*m+1:20)];
    ksplit_val_targ=[ktar_data(2*m-1);ktar_data(2*m)];
    
    w_star=inv((U*U'+lambda(u)*eye(10,10)))*U*ksplit_tr_targ;
    U3=[X(:,2*m-1) X(:,2*m)];
   %cost1=cost1+norm(U3'*w_star-ksplit_val_targ)^2+lambda(u)*norm(w_star)^2;
   cost1=cost1+norm(U3'*w_star-ksplit_val_targ)^2+lambda(u)*norm(w_star(2:10))^2;
  k_split_cost(u)=(2*cost1/20)^0.5;
[M2,I]=min(k_split_cost);
lambda_star=(I-1)*dx1;

endfor
endfor

subplot(2,3,2);
l=1:N_iter;
plot(log((l-1)*dx1),k_split_cost(l));
ylabel('RMS cost');
xlabel('log(lambda)');
title('10Fold CV');
%calculate test cost
test_inp=test(:,1);
test_targ=test(:,2);

X_test=zeros(10,size(test)(1));
for i =1:10
  for t =1:size(test)(1)
    X_test(i,t)=test(t)(1)^(i-1);
endfor
endfor

w_star_test=inv((X*X'+lambda_star*eye(10,10)))*X*ktar_data;
%cost_test=norm(X_test'*w_star_test-test_targ)^2+lambda_star*norm(w_star_test)^2;
cost_test=norm(X_test'*w_star_test-test_targ)^2+lambda_star*norm(w_star_test(2:10))^2;
%extra credit
%leave one out
lo_cost=zeros(N_iter,1);
U=zeros(10,20);
for i=1:10
  for j=1:20
    U(i,j)=inp_data(j)^(i-1);
endfor
endfor
for u =1:N_iter
  cost=0;
  for c=1:20
    lo_tr_inp=[inp_data(1:c-1);inp_data(c+1:20)];
    lo_tr_targ=[tar_data(1:c-1);tar_data(c+1:20)];
    lo_val_inp=[inp_data(c)];
    lo_val_targ=[tar_data(c)];
    U_tr=[U(:,1:c-1) U(:,c+1:20)];
    U_val=[U(:,c)];
    w_star_lo=inv((U_tr*U_tr'+lambda(u)*eye(10,10)))*U_tr*lo_tr_targ;
    %cost=cost+norm(U_val'*w_star_lo-lo_val_targ)^2+lambda(u)*norm(w_star_lo)^2;
    cost=cost+norm(U_val'*w_star_lo-lo_val_targ)^2+lambda(u)*norm(w_star_lo(2:10))^2;
  lo_cost(u)=(2*cost/10)^0.5;

endfor
endfor
[M3,I2]=min(lo_cost);
lo_lambda=(I2-1)*dx1;

lo_w_star=inv((U*U'+lo_lambda*eye(10,10)))*U*tar_data;
%lo_cost_test=norm(X_test'*lo_w_star-test_targ)^2+lo_lambda*norm(lo_w_star)^2;
lo_cost_test=norm(X_test'*lo_w_star-test_targ)^2+lo_lambda*norm(lo_w_star(2:10))^2;
l=1:N_iter;
subplot(2,3,3);
plot(log((l-1)*dx1),lo_cost(l));
ylabel('RMS cost');
xlabel('log(lambda)');
title('Leave One Out');
lambda_star=lambda_star
kfoldcost=(2*cost_test/size(test)(1))^0.5
simplecost=(2*cost_test_simple/size(test)(1))^0.5
leaveonecost=(2*lo_cost_test/size(test)(1))^0.5
%error bars
subplot(2,3,4)
for u =1:size(test(:,2))(1)
  simple(u)=(X_te(:,u)'*w_st-te_targ(u))^2;
endfor
errorbar(test(:,1),test(:,2),simple(:),'rd');
%errorbar((X_te'*w_st)(1:50),test(1:50,2),simple(1:50),'rd');
xlabel('Test input')
ylabel('Test outputs')
title('Variance for simple split')
subplot(2,3,5)
for u =1:size(test(:,2))(1)
  k_sp(u)=(X_test(:,u)'*w_star-te_targ(u))^2;
endfor
errorbar(test(:,1),test(:,2),k_sp(:),'rd');
%errorbar((X_test'*w_star)(1:50),test(1:50,2),k_sp(1:50),'rd');
xlabel('Test input')
ylabel('Test outputs')
title('variance with with 10-Fold')

subplot(2,3,6)

for u =1:size(test(:,2))(1)
  l_one(u)=(X_te(:,u)'*lo_w_star-te_targ(u))^2;
endfor
errorbar(test(:,1),test(:,2),l_one(:),'rd');
%errorbar((X_te'*lo_w_star)(1:50),test(1:50,2),l_one(1:50),'rd');
xlabel('Test input')
ylabel('Test outputs')
title('Variance with leave one out')