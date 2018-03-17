% this function takes train matrix, train target matrix, cv inp and tar matrix. and the best/test lambda
% it then calculates the weights matrix and computes the mean squared error on the cv set.

function error = computeError(train_inp,train_tar,cv_inp,cv_tar,lbda)

weights_matrix = pinv(train_inp*train_inp'+lbda*eye(size(train_inp,1)))*train_inp*train_tar;

error = immse(cv_inp'*weights_matrix,cv_tar+lbda*norm(weights_matrix));