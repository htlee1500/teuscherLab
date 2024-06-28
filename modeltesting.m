%n20 = readtable("20.csv");
n460 = readtable("25-460.csv");
n625 = readtable("460-625.csv");
n1446 = readtable("625-1446.csv");
n2101 = readtable("1446-2101.csv");
n3000 = readtable("2101-3000.csv");

N = n460.N;
Q = n460.Q;
f = fit(N, Q, 'poly4');
%plot(f, N, Q, 'residuals');

M = n625.N;
R = n625.Q;
g = fit(M, R, 'exp2');
%plot(g, M, R);

L = n1446.N;
S = n1446.Q;
h = fit(L, S, 'weibull');
%plot(h, L, S, 'residuals');

newtable = readtable("temp.csv");

A = newtable.N;
B = newtable.Q;
j = fit(A, B, 'power2');
plot(j, A, B, 'residuals');