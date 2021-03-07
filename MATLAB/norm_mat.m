function Y=norm_mat(X,varargin)
%% normalize the matrix to [0 , 1]. If the second varargin is True, then normalize that to [-1,1].

Y=(X-min(min(X)))./(max(max(X))-min(min(X)));
if(length(varargin) == 1 && varargin{1})
    Y = (Y - 0.5)*2;
end

end