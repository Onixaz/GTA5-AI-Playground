# GTA5-AI-Playground

My take on GTA5 self driving cars. 
I've commented the interesting parts of the code and tried make this as simple as possible. 
Some parts are still messy and code needs some cleaning done so free to change anything you like. 
This is pretty much only a cleaner version of Sendtex's early Pygta5 project https://github.com/Sentdex/pygta5 so credits to that amazing guy. 
Note that i'm using Keras instead of Tflearn and I added data augmentation with custom batch generator(thanks to Ryan Slim for the ideas) so you will probably need less data overall. 
Still I would aim for 100k+ samples.

To use my version: Just clone the repository, install required dependencies(I strongly suggest Anaconda), edit the scripts to set the paths for data(didnt bother with argparse, sorry). 
Start your GTA5 in windowed mode(800x600),if using manual screen grab drag the window to top left corner and then:

1. collect_data.py
2. split_data.py
3. train_model.py
4. test_model.py

Trained and tested on GTX 1080. 
If you're getting error 
"E tensorflow/stream_executor/cuda/cuda_dnn.cc:373] Could not create cudnn handle: CUDNN_STATUS_ALLOC_FAILED"
while trying to run test_model.py, alt+tab the GTA5 to taskbar while the script is loading. 

Again for more info and tutorials refer to https://github.com/Sentdex/pygta5 or open the issue.
