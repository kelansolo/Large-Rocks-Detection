## How to run a job on the SCITAS machines

For the most part, follow the instructions from the [Moodle page]([url](https://moodle.epfl.ch/mod/resource/view.php?id=1226324)). If you want to test thing in a jupyter notebook, simply follow the PDF.

To run a batch (this is likely what we will do towards the end when we really want to train big models on a lot of data):

Create a `.run` file with the following structure
```
!/bin/bash

#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBAtCH --mem 16G
#SBATCH --time 1:00:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

# Your commands
```

Then at the end of the file, at the commands that you want to run. This could directly be a call to train the model `yolo train ...`  or, and that`s probably the easiest, a python file `python3 myfile.py <args>`. You can then train and validate the model inside the python file.
