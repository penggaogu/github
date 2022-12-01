%% Steepest Descent Method
%% Input information
x = [0;
     0];
tol = 0.05;
max_iter = 15;
J = @(x) [8*x(1)-20, 1/2*x(2); 1/2*x(2)^2+2, x(1)*x(2)-5];
F = @(x) [4*x(1)^2-20*x(1)+1/4*x(2)^2+8; 1/2*x(1)*x(2)^2+2*x(1)-5*x(2)+8];
f1 = @(x) 4*x(1)^2-20*x(1)+1/4*x(2)^2+8;
f2 = @(x) 1/2*x(1)*x(2)^2+2*x(1)-5*x(2)+8;
g = @(x) f1(x)^2 + f2(x)^2 ;
%% Steepest Descent Method
k = 1;
fprintf('k\t\tx1\t\tx2\t\tg \n')
fprintf('%d\t%f\t%f\n',0,x(1),x(2))
while( k <= max_iter)
    g1 = g(x);
    z = 2*transpose(J(x))*F(x);
    z0 = norm(z);
    if(z0 == 0)
        fprintf('Zero gradient.')
        fprintf('%d\t%,9f\t%,9f\t%.9f\n',k,x(1),x(2),g1)
        break;
    end
    z = z/z0;
    alpha1 = 0;
    alpha3 = 1;
    g3 = g(x - alpha3*z);
    while(g3 >= g1)
        alpha3 = alpha3/2;
        g3 = g(x - alpha3*z);
        if(alpha3 < tol/2)
            fprintf('No likely improvement.\n')
            break;
        end
    end
    alpha2 = alpha3/2;
    g2 = g(x - alpha2*z);
    h1 = (g2-g1)/alpha2;
    h2 = (g3-g2)/(alpha3-alpha2);
    h3 = (h2-h1)/alpha3;
    alpha0 = 0.5*(alpha2 - h1/h3);
    g0 = g(x - alpha0*z);
    if(g0 <= g3)
        g_min = g0;
        alpha = alpha0;
    else
        g_min = g3;
        alpha = alpha3;
    end
    x = x-alpha*z;
    fprintf('%d\t%f\t%f\t%f\n',k,x(1),x(2),g_min);
    if(abs(g_min-g1)<tol)
        break;
    end
    k = k+1;
end







