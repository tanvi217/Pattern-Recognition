clc
clear
close all

%%

N = 2000;
x = 15+randn(N,2);

h= 15;
k=15;
r=7;
t=(1:N)*2*pi/N;
t= t(:);
plot( r*cos(t)+h, r*sin(t)+k);

figure;
y = [r*cos(t)+h+1*randn(N,1),r*sin(t)+k+1*randn(N,1)];
plot(x(:,1),x(:,2),'b>',y(:,1),y(:,2),'mo','LineWidth',1.5,'MarkerFaceColor','w');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Class 1','Class 2');

Z1(1:2:2*N-1,:) = x;
Z1(2:2:2*N,:) = y;
TestTarg1(1:2:2*N-1) = 1;
TestTarg1(2:2:2*N) = -1;
 
toroiddata = [Z1 TestTarg1(:)];
trainingdata = toroiddata;
vx = var(x);
vy = var(y);


figure;
lx = sqrt(sum((abs(x-ones(size(x,1),1)*(mean(Z1)))').^2));
ly = sqrt(sum((abs(y-ones(size(y,1),1)*(mean(Z1)))').^2));
plot(lx,'b>','LineWidth',1.5,'MarkerFaceColor','w');
hold on;
plot(ly,'mo','LineWidth',1.5,'MarkerFaceColor','w');
xlabel('Distance');
ylabel('Features');
legend('Class 1','Class 2');

T = TestTarg1;
PhiZ = sqrt(sum(abs((Z1-ones(size(Z1,1),1)*(mean(Z1)))').^2));
Xmat = [ones(size(Z1,1),1) PhiZ(:)];

W_ls  = regress(T(:),Xmat);
Y_x = Xmat*W_ls;
thr = -W_ls(1)/W_ls(2)

pred_labels = ones(size(T));
pred_labels(logsig(Y_x) < 0.5) = 2;
T(T == -1) = 2;

figure;
plot(lx,'b>','LineWidth',1.5,'MarkerSize',10,'MarkerFaceColor','w');
hold on;
plot(ly,'mo','LineWidth',1.5,'MarkerSize',10,'MarkerFaceColor','w');
plot(PhiZ(pred_labels == 1),'k+','LineWidth',1.5,'MarkerFaceColor','w');
plot(PhiZ(pred_labels == 2),'y*','LineWidth',1.5,'MarkerFaceColor','w');
plot(thr*ones(N),'r','LineWidth',2);
hold off;


ConfMat = ConfusionMatrix2(pred_labels,T,2);
disp(ConfMat);
acc = sum(diag(ConfMat))/sum(sum(ConfMat));
disp(acc);

mu = mean(Z1);
z1vec = (mu(1)-thr):0.01:(mu(1)+thr);
ix = 1;
for zx = 1:length(z1vec)
    z1 = z1vec(zx);
    z2 = mu(2)+sqrt(thr.^2 - abs(z1-mu(1)).^2);
    model(ix,:) = [z1,z2];
    ix = ix + 1;
    z2 = mu(2)-sqrt(thr.^2 - abs(z1-mu(1)).^2);
    model(ix,:) = [z1,z2];
    ix = ix+1;
end
figure;
plot(x(:,1),x(:,2),'b>',y(:,1),y(:,2),'mo','LineWidth',1.5,'MarkerFaceColor','w');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Class 1','Class 2');
 
hold on;
plot(model(:,1),model(:,2),'g.','LineWidth',2);
