% file name:  steepdesc.m
% This Matlab code implements Cauchy's steepest descent method
% using Armijo stepsize rule.
% It terminates when the norm of the gradient is below 10^(-6).
% The function value is read from the file "func.m".
% The gradient value is read from the file "grad.m".

% Read in inputs
  n=input('enter the number of variables n ');
  x=input('enter the initial column vector x ');

% Armijo stepsize rule parameters
  sigma = .1;
  beta = .5;
  obj=func(x);
  g=gradient(x);
  k=0;                                  % k = # iterations
  nf=1;					% nf = # function eval.	

% Begin method
  while  norm(g) > 1e-6    
    d = -g;                   % steepest descent direction
    a = 1;
    newobj = func(x + a*d);
    nf = nf+1;
    while (newobj-obj)/a > sigma*g'*d
      a = a*beta;
      newobj = func(x + a*d);
      nf = nf+1;
    end
    if (mod(k,100)==1) fprintf('%5.0f %5.0f %12.5e \n',k,nf,obj); end
    x = x + a*d;
    obj=newobj;
    g=gradient(x);
    k = k + 1;
  end

% Output x and k
  x, k

x

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%file name:  func.m
%This routine evaluates the Rosenbrock function.

function y = func(x)
y = 100*(x(1)^2 - x(2))^2 + (x(1)-1)^2;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%file name:  grad.m
%This routine evaluates the gradient of the Rosenbrock function

function y = gradient(x)
y(1) = 100*(2*(x(1)^2-x(2))*2*x(1)) + 2*(x(1)-1);
y(2) = 100*(-2*(x(1)^2-x(2)));
y = y';

