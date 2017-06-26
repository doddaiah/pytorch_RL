1. Enable render openAI gym in server

* Reinstall Nvidia driver and cuda library with -no-opengl-files (ref:https://davidsanwald.github.io/2016/11/13/building-tensorflow-with-gpu-support.html)
> test: glxinfo

* Install VNC server and viewer

* Others: VirtualGL

2. @property decorator will make the function be called occasionally, even withouting going into the function body. So be careful that you want to set a property only when you need it to be attribute-like property

3. Python3.6 have wield routine of calling base construtor. So anything you make different from base is after calling super().\__init__

4. Add dt in OU process, which is required in OU process equations.

5. __Don't use for loop to apply hidden layer in forward function__, because
   otherwise all hidden units weights are the same, which would limit the
   starting point the network. The for loop should be use in RNN, where weights
   and modules are required to be the same.
