disp('hi!!!')


clear all
close all
clc

temp = load('data_train.mat');
data_train = temp.data_train;
temp = load('data_test.mat');
data_test = temp.data_test;
temp = load('labels_train.mat');
labels_train = temp.labels_train;
temp = load('labels_test.mat');
labels_test = temp.labels_test;

cd ..
pwd



