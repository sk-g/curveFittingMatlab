function loss = computeCost(lbda,inp,target)
loss = zeros(1,length(lbda));
for i = 1:length(lbda)
	w = inv(inp*inp'+lbda(i)*eye(size(inp,1)))*inp*target;
	loss(1,i) = norm(inp'*w-target,2) + lbda(i)*norm(w,2);
endfor