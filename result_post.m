clear;
clc;
load ../XXXX.test

val_name = who;
eval(char(strcat('conf_mat = ', val_name, ';')));

test_count = sum(sum(conf_mat));
corr_count = sum(diag(conf_mat));

%display accuracy
temp_str = ['accuracy: ', num2str(corr_count/test_count*100), '%'];
display(sprintf(temp_str));

[class_count, ~] = size(conf_mat);

for class_counter = 1:class_count,
    temp_str = ['class ', num2str(class_counter), ': '];
    display(sprintf(temp_str));
    %construct TF conf_mat
    TP = conf_mat(class_counter, class_counter);
    FN = sum(conf_mat(class_counter,:)) - TP;
    FP = sum(conf_mat(:, class_counter)) - TP;
    TN = test_count - TP - FN - FP;
    %display sensitivity
    temp_str = ['sensitivity: ', num2str(TP/(TP + FN)*100), '%'];
    display(sprintf(temp_str));
    %display specificity
    temp_str = ['specificity: ', num2str(TN/(FP + TN)*100), '%'];
    display(sprintf(temp_str));
    %display precision
    prec = TP/(TP + FP);
    temp_str = ['precision: ', num2str(prec*100), '%'];
    display(sprintf(temp_str));
    %display recall
    reca = TP/(TP + FN);
    temp_str = ['recall: ', num2str(reca*100), '%'];
    display(sprintf(temp_str));
    %display F1
    beta = 1;
    temp_str = ['F1: ', num2str(((beta^2 + 1)*prec*reca)/(beta^2*prec + reca)*100), '%'];
    display(sprintf(temp_str));
    %display F05
    beta = 0.5;
    temp_str = ['F0.5: ', num2str(((beta^2 + 1)*prec*reca)/(beta^2*prec + reca)*100), '%'];
    display(sprintf(temp_str));
    %display F2
    beta = 2;
    temp_str = ['F2: ', num2str(((beta^2 + 1)*prec*reca)/(beta^2*prec + reca)*100), '%', '\n'];
    display(sprintf(temp_str));
    display(sprintf('\n'));
end;
