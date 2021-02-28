# Section 2: Training a segmentation CNN

You will be using PyTorch to train the model, similar to exercises in the Segmentation & Classification Lesson, and we will be using Tensorboard to visualize the results.

You will use the script `run_ml_pipeline.py` to kick off your training pipeline.

The code has hooks to log progress to Tensorboard. In order to see the Tensorboard output you need to launch Tensorboard executable from the same directory where run_ml_pipeline.py is located using the following command:

`tensorboard --logdir runs --bind_all`

After that, Tensorboard will write logs into directory called runs and you will be able to view progress by opening the browser and navigating to default port 6006 of the machine where you are running it.

## Tensorboard
1. Make sure you enable GPU.
2. In a terminal move to the src directory through `cd src`
3. Then run the following command to start tensorboard: `tensorboard --logdir runs --bind_all`
4. The output should have a URL
5. Copy that URL
6. Open the Desktop with the Desktop button at the bottom right hand corner and copy the URL. (It will look something like http://f8196ac7f2cc:6006/)
7. If not already open, open a browser.
8. On the left hand side there will be an arrow, if you click the arrow, a sidebar that will popup.
9. Click on the 2nd icon on the sidebar which is a clipboard and paste the URL here.
10. In the address bar you can right click and select Paste & Go.
