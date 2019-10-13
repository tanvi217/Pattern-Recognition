clc
clear
close all

%%

N = 2000;
h= 15;
k=15;
t=linspace(-3.5,3.5,2000);
t= t(:);

x = [5+0.75*cosh(t)+1*randn(N,1),0.75*sinh(t)+1*randn(N,1)];
x1=[-(5+0.75*cosh(t)+1*randn(N,1)),0.75*sinh(t)+1*randn(N,1)];
X=[x;x1];
y = [0.5*sinh(t)+1*randn(N,1),5+0.5*cosh(t)+1*randn(N,1)];
y1=[0.5*sinh(t)+1*randn(N,1),-(5+0.5*cosh(t)+1*randn(N,1))];
Y=[y;y1];

plot(X(:,1),X(:,2),'b>',Y(:,1),Y(:,2),'mo','LineWidth',1.5,'MarkerFaceColor','w');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Class 1','Class 2');

Z1(1:2:4*N-1,:) = X;
Z1(2:2:4*N,:) = Y;
TestTarg1(1:2:4*N-1) = 1;
TestTarg1(2:2:4*N) = -1;
trainingdata=[Z1 TestTarg1(:)];

%-------------
disp("Linear Discriminant");

[trainedclass,tenfold]=lineardiscriminant(trainingdata);

pred_labels = trainedclass.predictFcn(Z1);
T = TestTarg1;
T(T==-1)=2;
ConfMat = ConfusionMatrix2(pred_labels,T,2)

f1metrics=MyClassifyPerf(TestTarg1,pred_labels)
figure;
plot(X(:,1),X(:,2),'b>','LineWidth',1.5,'MarkerSize',10,'MarkerFaceColor','w');
hold on;
plot(Y(:,1),Y(:,2),'mo','LineWidth',1.5,'MarkerSize',10,'MarkerFaceColor','w');
plot(Z1(pred_labels == 1,1),Z1(pred_labels == 1,2),'k+','LineWidth',1.5,'MarkerFaceColor','w');

plot(Z1(pred_labels == -1,1),Z1(pred_labels == -1,2),'y*','LineWidth',1.5,'MarkerFaceColor','w');
legend("class1 actual","class2 actual","predicted as class1","predicted as class2");
title("Linear Discriminant Model");

Z1range = min(Z1(:,1)):.01:max(Z1(:,1));
Z2range = min(Z1(:,2)):.01:max(Z1(:,2));
[xx1, xx2] = meshgrid(Z1range,Z2range);
ZG= [xx1(:) xx2(:)];
predictedspecies = trainedclass.predictFcn(ZG);
figure;
gscatter(xx1(:), xx2(:), predictedspecies,'cy');
hold on;
plot(X(:,1),X(:,2),'b>',Y(:,1),Y(:,2),'mo','LineWidth',0.2,'MarkerFaceColor','w');
title("Linear Discriminant Model");
xlabel("Feature 1");
ylabel("Feature 2");
disp(tenfold);


%--------------
disp("Quadratic discrimanant")
[trainedclass,tenfold]=quadraticdiscrimanant(trainingdata);

pred_labels = trainedclass.predictFcn(Z1);
T = TestTarg1;
T(T==-1)=2;
ConfMat = ConfusionMatrix2(pred_labels,T,2)

f1metrics=MyClassifyPerf(TestTarg1,pred_labels)
figure;
plot(X(:,1),X(:,2),'b>','LineWidth',1.5,'MarkerSize',10,'MarkerFaceColor','w');
hold on;
plot(Y(:,1),Y(:,2),'mo','LineWidth',1.5,'MarkerSize',10,'MarkerFaceColor','w');
plot(Z1(pred_labels == 1,1),Z1(pred_labels == 1,2),'k+','LineWidth',1.5,'MarkerFaceColor','w');
plot(Z1(pred_labels == -1,1),Z1(pred_labels == -1,2),'y*','LineWidth',1.5,'MarkerFaceColor','w');
legend("class1 actual","class2 actual","predicted as class1","predicted as class2");
title("Quadratic Discriminant Model");


Z1range = min(Z1(:,1)):.01:max(Z1(:,1));
Z2range = min(Z1(:,2)):.01:max(Z1(:,2));
[xx1, xx2] = meshgrid(Z1range,Z2range);
ZG= [xx1(:) xx2(:)];
predictedspecies = trainedclass.predictFcn(ZG);
figure;
gscatter(xx1(:), xx2(:), predictedspecies,'cy');
hold on;
plot(X(:,1),X(:,2),'b>',Y(:,1),Y(:,2),'mo','LineWidth',0.2,'MarkerFaceColor','w');
title("Quadratic Discriminant Model");
xlabel("Feature 1");
ylabel("Feature 2");
disp(tenfold);


%-----------
disp("Guassian SVM")
[trainedclass,tenfold]=gaussiansvm(trainingdata);

pred_labels = trainedclass.predictFcn(Z1);
T = TestTarg1;
T(T==-1)=2;
ConfMat = ConfusionMatrix2(pred_labels,T,2)

f1metrics=MyClassifyPerf(TestTarg1,pred_labels)
figure;
plot(X(:,1),X(:,2),'b>','LineWidth',1.5,'MarkerSize',10,'MarkerFaceColor','w');
hold on;
plot(Y(:,1),Y(:,2),'mo','LineWidth',1.5,'MarkerSize',10,'MarkerFaceColor','w');
plot(Z1(pred_labels == 1,1),Z1(pred_labels == 1,2),'k+','LineWidth',1.5,'MarkerFaceColor','w');

plot(Z1(pred_labels == -1,1),Z1(pred_labels == -1,2),'y*','LineWidth',1.5,'MarkerFaceColor','w');
legend("class1 actual","class2 actual","predicted as class1","predicted as class2");
title("SVM with Gaussian Kernel");


Z1range = min(Z1(:,1)):.01:max(Z1(:,1));
Z2range = min(Z1(:,2)):.01:max(Z1(:,2));
[xx1, xx2] = meshgrid(Z1range,Z2range);
ZG= [xx1(:) xx2(:)];
predictedspecies = trainedclass.predictFcn(ZG);
figure;
gscatter(xx1(:), xx2(:), predictedspecies,'cy');
hold on;
plot(X(:,1),X(:,2),'b>',Y(:,1),Y(:,2),'mo','LineWidth',0.2,'MarkerFaceColor','w');
title("SVM with Gaussian Kernel");
xlabel("Feature 1");
ylabel("Feature 2");
disp(tenfold);

%-----------
disp("Quadratic SVM")
[trainedclass,tenfold]=quadraticsvm(trainingdata);

pred_labels = trainedclass.predictFcn(Z1);
T = TestTarg1;
T(T==-1)=2;
ConfMat = ConfusionMatrix2(pred_labels,T,2)

f1metrics=MyClassifyPerf(TestTarg1,pred_labels)
figure;
plot(X(:,1),X(:,2),'b>','LineWidth',1.5,'MarkerSize',10,'MarkerFaceColor','w');
hold on;
plot(Y(:,1),Y(:,2),'mo','LineWidth',1.5,'MarkerSize',10,'MarkerFaceColor','w');
plot(Z1(pred_labels == 1,1),Z1(pred_labels == 1,2),'k+','LineWidth',1.5,'MarkerFaceColor','w');

plot(Z1(pred_labels == -1,1),Z1(pred_labels == -1,2),'y*','LineWidth',1.5,'MarkerFaceColor','w');
legend("class1 actual","class2 actual","predicted as class1","predicted as class2");
title("SVM with Quadratic Kernel");


Z1range = min(Z1(:,1)):.01:max(Z1(:,1));
Z2range = min(Z1(:,2)):.01:max(Z1(:,2));
[xx1, xx2] = meshgrid(Z1range,Z2range);
ZG= [xx1(:) xx2(:)];
predictedspecies = trainedclass.predictFcn(ZG);
figure;
gscatter(xx1(:), xx2(:), predictedspecies,'cy');
hold on;
plot(X(:,1),X(:,2),'b>',Y(:,1),Y(:,2),'mo','LineWidth',0.2,'MarkerFaceColor','w');
title("SVM with Quadratic Kernel");
xlabel("Feature 1");
ylabel("Feature 2");
disp(tenfold);

%-----------
disp("KNN")
[trainedclass,tenfold]=knn(trainingdata);

pred_labels = trainedclass.predictFcn(Z1);
T = TestTarg1;
T(T==-1)=2;
ConfMat = ConfusionMatrix2(pred_labels,T,2)

f1metrics=MyClassifyPerf(TestTarg1,pred_labels)
figure;
plot(X(:,1),X(:,2),'b>','LineWidth',1.5,'MarkerSize',10,'MarkerFaceColor','w');
hold on;
plot(Y(:,1),Y(:,2),'mo','LineWidth',1.5,'MarkerSize',10,'MarkerFaceColor','w');
plot(Z1(pred_labels == 1,1),Z1(pred_labels == 1,2),'k+','LineWidth',1.5,'MarkerFaceColor','w');

plot(Z1(pred_labels == -1,1),Z1(pred_labels == -1,2),'y*','LineWidth',1.5,'MarkerFaceColor','w');
legend("class1 actual","class2 actual","predicted as class1","predicted as class2");
title("K-Nearest Neighbour Classifier");

Z1range = min(Z1(:,1)):.01:max(Z1(:,1));
Z2range = min(Z1(:,2)):.01:max(Z1(:,2));
[xx1, xx2] = meshgrid(Z1range,Z2range);
ZG= [xx1(:) xx2(:)];
predictedspecies = trainedclass.predictFcn(ZG);
figure;
gscatter(xx1(:), xx2(:), predictedspecies,'cy');
hold on;
plot(X(:,1),X(:,2),'b>',Y(:,1),Y(:,2),'mo','LineWidth',0.2,'MarkerFaceColor','w');
title("K-Nearest Neighbour Classifier");
xlabel("Feature 1");
ylabel("Feature 2");
disp(tenfold);
