# Shape and Colour detection using OpenCV

Scope: The main goal of this project was to create a python script that could identify the shape and colour of objects shown to a webcam. 
Action: To begin this project, I first had to research different approaches for distinguishing shapes and colours. As the objects were of a pre-known set of colours, an option was to use a colour mask to identify a set of shapes of that colour, and then separate them out using contour generation. An additional option was to create a fixed morphology and use contouring to find every shape. I decided for go for the latter option as this opened up more options, and the code would not have to change if an additional colour was added to the set of objects. 
Result: Most issues arose when finding which functions were best suited for each functionality as there are a wide variety of prebuild modules for each application, researching each and finding the benefits/drawbacks of each methodology was key in finding a good fit.

## Installation

To run this code you must have the OpenCV lirary install to your python distribution. 

### Anaconda Command Line:
```bash
conda install -c conda-forge opencv
```
### Windows Command Line:
```bash
pip install opencv-python
```

## Usage

Simply run "Final version.py" and type either img to select an image from your files, or type cam to get a live preview of your webcam, with detected shapes being outlines and named on either.

Alternatively, run the "MLP Classifier Training.py" and then the "Shape and Colour Recognition.py" scripts. This will allow for the training of the classifier to be executed separately.

If you are having issues displaying an image or are getting an error, you may need to change the "1" in line 21 to a "0":
```python
cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
                       ^
```

Once the preview is shown, there will be a trackbar that allows you to manually adjust the exposure of your webcam.

To exit the live preview, please ensure the window containing the image is the focus window and press "esc".


## Access to Training Images
[Here](https://drive.google.com/drive/folders/11hFxIGERfd7P18XZfk2TZ-y2D_LYZMV9?usp=sharing) is a link to a folder containing the images used to train the neural network for detecting shapes.
