function mret = mat2vet(q0)
%Coloca a matriz em um vetor, por linhas
q = q0';
n = size(q);
if ~iscell(q0)
    x = zeros(1,prod(n));
    x(1:prod(n)) = q(q | ~q)';
else
    qmat = ones(n(1),n(2));
    x = q(qmat==1)';
end    
mret = x;