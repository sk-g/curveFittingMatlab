load train.txt
load test.txt
% loading into x and target (t)

inp_col = train(:,1);

tar_col = train(:,2);
t = tar_col;

X = zeros(10,20);
for i = 1:10
	for j = 1:20
		X(i,j) = inp_col(j)^(i-1);
	endfor
endfor
testset_X = zeros(10,size(test,1));
for s = 1:10
	for p = 1:size(test,1)
		testset_X(s,p) = test(s,1)^(p-1);
	endfor
endfor

% lambda parameter values

xStart = -0.001;
dx1 = 0.0003/15;
N_iter = 3500;
lbda = xStart + (0:N_iter-1)*dx1;
%lbda = [0.00000001 0.000001 0.0001 0.001 0.01 0.05 0.09 0.1 1 10 20 30 50 80 100];
%lbda = [ 0.0001 0.001 0.01 0.1 1 10 100 1000]

%simple split :
% 70% train and 30% validation %

X_train = X(1:10,1:14);
X_test = t(1:14);


loss = zeros(1,length(lbda));


loss_on_complete_train = computeCost(lbda,X,t);

temp = computeCost(lbda,X_train,X_test);


[lowest_loss, lowest_loss_idx] = min(temp);
best_lbda = lbda(lowest_loss_idx)
loss_from_func = computeError(X_train,X_test,X(1:10,15:20),t(15:20),best_lbda)

plot((lbda),loss_on_complete_train,'-rd',(lbda),temp,'-bd');
legend('Complete train set','Split data', "location", "northwest");
xlabel("(lambda)");
ylabel("loss");

