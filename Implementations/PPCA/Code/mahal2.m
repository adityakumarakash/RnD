function d=mahal2(Y, X)

% mahalanobis distance function

[rx,cx] = size(X);
[ry,cy] = size(Y);

if cx ~= cy
   error(message('stats:mahal:InputSizeMismatch'));
end


m = mean(X,1);
M = m(ones(ry,1),:);

C = X - m(ones(rx,1),:);
[Q,R] = qr(C,0);

ri = R'\(Y-M)';
d = sum(ri.*ri,1)'*(rx-1);