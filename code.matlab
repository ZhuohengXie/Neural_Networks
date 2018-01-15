>> seeds_data=importdata('seeds_data.txt')
%%import the origin dataset
>> seeds_data=seeds_data'
%%transpose the origin dataset
>> x=importdata('seeds_x.txt')
%%importdata to get input matrix
>> t=importdata('seeds_t.txt')
%%importdata to get target matrix
>> size(x)
%%view the size of inputs x, 7 rows, 210 columns
>> size(t)
%%view the size of target t, 3 rows, 210 columns
>> net = feedforwardnet(10);
%%build the feedforward network with 10 neurons in single hidden layer. Also tried 1 to
30.
>> view(net)
%%view the network
>> [net,tr] = train(net,x,t);
%%train the network
>> plotperform(tr)
%%plot performance graph to see how network’s performance improved during training,
to see the mean squared error (MSE) of the trained neural network
>> testX = x(:,tr.testInd);
%%generate test set
>> testT = t(:,tr.testInd);
%%generate test set
>> testY = net(testX);
%%test the network
>> testIndices = vec2ind(testY)
%%use vec2ind function to get the class indices as the position of the highest element in
each output vector
>> plotconfusion(testT,testY)
%% plot confusion matrix to see the performance
>> [c,cm] = confusion(testT,testY);
>> fprintf('Percentage Correct Classification : %f%%\n', 100*(1-c));
%% print the accuracy percentage
>> plotroc(testT,testY) %%plot ROC graph to see the performance
%% generate ROC graph