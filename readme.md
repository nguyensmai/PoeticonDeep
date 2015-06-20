% Copyright (c) 2014 Sao Mai Nguyen

%               e-mail : nguyensmai@gmail.com

%               http://nguyensmai.free.fr/

For a longer tutorial, please refer to readme.html

Quick Tutorial
==============
Requirements
--------------
The source code is accesible via git from https://github.com/nguyensmai/PoeticonDeep.git
Matlab with Optimisation and Statistics Toolbox

Testing the mnist dataset
--------------
execute file `DmpDeepBelief/mnist/minst_generativeModel_small.m`
This is a sample program with a small neural network (few nodes at each layer) so you will probably see progress in learning, but still high error rates.

To see better results, execute file `DmpDeepBelief/mnist/minst_generativeModel.m`. This is the same as the other file, but with a larger network (more nodes, so slower training).

Summary
==============
To read the model, refer to [[Poeticon]] articles, and more specifically  [[Deep Belief Network Architecture for Poeticon ++]] proposes the latest  model and architecture.

The architecture relies on [[Deep learning]], and more specifically Deep Boltzmann Machines. A  detailed description of the actual code with the different files and folders  can be read in [[Description of the files]].


We also explored several methods for encoding movements, for ex [[Dynamic Movement Primitives|DMP]]. 



Folders organisation
==============
The organisation now is messy, and should be improved once a working version has been tested.

Currently, the code contains the following folders:

- ``Autoencode_Code``: it is a modified version of the DBN code from hinton's website http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html
- ``dmp_bbo_matlab_deprecated-master_deprecated`` : code for the Dynamic Motor Primitives as it was downloaded from https://github.com/stulp/dmp_bbo_matlab_deprecated. This library is well-written, every function has a unit test function to see how it works. It is also available in C/C++.
- ``DmpDeepBelief/mnist`` is a modified version of the DBM code from [[http://www.cs.toronto.edu/~rsalakhu/DBM.html]].
- ``DmpDeepBelief/writeNumbersDmp`` is the version of the DBM for learning the DMP  parameters for writing digits.


Notable files
==============
The main files
--------------
- For mnist, the main file is ``minst_generativeModel.m``. It creates batches from the dataset of digit images, and pretrains them layer by layer, then finie-tunes the whole DBM with mean-field. Finally, it draws images corresponding to each label we ask it. 
- For writeNumbersDmp, the main file is ``writeNumbers.m``. It does the same thing as mnist_generativeModel.m, except that the dataset is different. It now uses the DMP parameters for writing the digits.

Important files
--------------

**Folder ~DmpDeepBelief/writeNumbersDmp**
- The class ``Cursor`` in file cursor.m for capturing the trajectory of the mouse when you write a digit.
- ``writeNumbersDmp.m`` then trains the DBM.


**Folder ~dmp_bbo_matlab_deprecated-master_deprecated**
- ``dmptrain`` takes as input movement trajectories and outputs the DMP parameters.
- ``dmpintegrate`` on the contrary takes as input the DMP parameters and outputs the movement trajectories.


**Folder Autoencoder_Code**
Has been originally downloaded from HInton's webpage.

- ``rbmsigmoid.m`` pretains with contrastive divergence a RBM with binary inputs and binary hidden layer. 
- ``rbmgaussian.m`` pretains with contrastive divergence a RBM with real-valued inputs and binary hidden layer. I had it working on an example, but it is reputed to be hard to train, therefore it might be wiser to avoid it. The easiest method is to still use rbmsigmoid but use the real valued inputs as probabilities to compute binary inputs.
- ``minstdeepauto`` provides a working example on the minst dataset.

**Folder WriteNumbers**
- `Cursor.m` provides with a graphical interface to capture mouse movements. First type a digit to represent the class the movement belongs to, then type 's' to start recording a movement, then 'e' to end the recording. You can continue recording other movements with 's' and 'e' keys. Finally press 'q' to end the program.
- `randDBN` are functions to create and initialise DBN/DBM.
-  `randRBM` and `assocRBM` are functions to create and initalise RBM : either a classic visible-hidden RBM or a (visible,label)-hidden RBM.
-  `generativeModel` takes as input labels and plots the corresponding data.
