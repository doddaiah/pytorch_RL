1. For episodic environment, the bellman backup for the final step should be the
   reward, instead of the sum of reward and one-step value function backup.  

   The reason may be that after a few iterations, all value functions should have
   reasonable values, and are ideally greater than 0. If we use one-step value
   function, this will over-estimate value function and error will propagate
   backwards, therefore before converging correctly, the over-estimation will
   impact value function approximation severely and lead to divergence. 

   ref: 99688077072ebf5e56179e78ec2d0e7ba13a794f

2. A `Variable` created by `torch.Tensor.cuda()` will still on cpu, if it is not
   directly used. For example, if it is directly fed into neural network, then
   it is cuda data, while if it is assigned to another variable, then it is cpu
   data

   ref: c7a0816a0ca1dd0303dcbd701984588632dfaa30
