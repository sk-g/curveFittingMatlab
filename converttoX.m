function converted = converttoX(X)
converted = zeros(size(X));
for i = 1:size(X,1)
	for j = 1:size(X,2)
		converted(i,j) = X(i)^(j-1);
	endfor
endfor