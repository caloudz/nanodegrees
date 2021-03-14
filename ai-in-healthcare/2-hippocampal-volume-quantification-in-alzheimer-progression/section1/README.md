# Section 1: Curating a Dataset of Brain MRIs

You will perform this in the section Workspace for Section 1. This workspace has a Python virtual environment called `medai` which is set up with everything that you need to train your ML model.

The data is located in `/data/TrainingSet` directory.

In the project directory called `section1` you will find a Python Notebook that has a few instructions in it that will help you inspect the dataset, understand the clinical side of the problem a bit better, and get it ready for consumption by your algorithm in Section 2. The notebook has 2 types of comments:
* Comments marked with `# TASK:` are tasks, instructions, or questions you have to complete.
* Comments not marked are not mandatory but are suggestions, questions, or background that will help you get a better understanding of the subject and apply your newly acquired medical imaging dataset EDA skills.

## Instructions

Once you complete the tasks, copy the following to the directory section1/out:
* Curated dataset with labels, as collection of NIFTI files. Amount of training image volumes should be the same as the amount of label volumes.
* A Python Notebook with the results of your Exploratory Data Analysis. If you prefer to do it in raw Python file, that is also acceptable - in that case put the .py file there, making sure that you copy Task descriptions from the notebook into your comments.
