# Development log 
update .gitignore to ignore model and log data 
## utils/cifar_util.py
- correct data size to original 50k
- normalise data using calculated channel mean and standard deviation 
## utils/ops.py
- add dropout operation 
- add scale parameter for batch normalisation 


## main.py 
- add automatic restoring from latest checkpoint 
- add a set of parameters for network.py to use 
- change experiment structure to 

		python main.py --model_id xxx
		
The models will be saved in modeldir/xxx together with current main.py, network.py and ops.py. 

## network.py
- add learning rate placeholder for scheduled learning rate 
- add L2 regularisation 
- add several layers to make the network standard VGG16-D
- add dropout options 
- add one more fully connected layer 
- change last activation to softmax
- add scheduled learning rate 
- Data augmentation 
	- add random flip 
	- add random crop 
- Testing 
	- add test all options, will test all checkpoints in of the given model
	- add debug information if a single checkpoint is tested 


