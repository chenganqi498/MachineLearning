function dist2 = distsq(a,b)
%c = rows of a.*a' + cols of b.*b'
c = bsxfun(@plus,sum((a.*a)')',sum((b.*b)'));
%c = c -2a*b
dist2 = c-2*a*b';