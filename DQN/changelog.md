1. For episodic environment, the bellman backup for the final step should be the
   reward, instead of the sum of reward and one-step value function backup.  

   The reason may be that after a few iterations, all value functions should have
   reasonable values, and are ideally greater than 0. If we use one-step value
   function, this will over-estimate value function and error will propagate
   backwards, therefore before converging correctly, the over-estimation will
   impact value function approximation severely and lead to divergence. 
