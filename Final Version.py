# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 11:09:16 2020

@author: tomwebb
"""

import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
import random
import tkinter as tk
from tkinter import filedialog

def na(x): #creates a function that can pass a null value as a variable, only used for trackbar to change exposure live
    pass

cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)   # set camera

cv2.namedWindow("Trackbars")    # create trackbar window
cv2.createTrackbar("Exposure", "Trackbars", 0, 10, na)  # create trackbar values


# Initialise list variables
features = []
features_shuffled_train_list = []
target_shuffled_train_list = []
train_list = []
features_shuffled_test_list = []
target_shuffled_test_list = []
shapes = ["A", "B", "C", "D", "E", "F"]
shape_class_list=["Cylinder"]*20+["Cube"]*20+["Triangular Prism"]*20+["Short Cuboid"]*20+["Long Cuboid"]*20+["Arch"]*20




# Large function used to complete morphology and other image manipulations, and then find the features from the contours found
def find_shapes(img, inp_image):
    list_of_shapes = []
    position_list = []
    box_list = []
    box_coord_list = []
    img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # converts image to grey
    img_sm = cv2.blur(img_grey, (5, 5))         # smooths image (lowers resolution)
    thr_value, img_th = cv2.threshold(img_sm, 110, 255, cv2.THRESH_BINARY)     # thresholds image to get either black or white
    kernel = np.ones((1,1),np.uint8)
    img_close = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, kernel)      # morphology correction
    img_canny = cv2.Canny(img_close, 60, 100)                          # edge detection
    contours, hierarchy = cv2.findContours(img_canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)   #finds contours and hierarchy from edge detection image
    
    for i, c in enumerate(contours):         # loop through all the found contours
        position_list.append((c[0,0,0],c[0,0,1]))
        epsilon = 0.05*cv2.arcLength(c, True)   #sets the epsilon value
        vertex_approx = len(cv2.approxPolyDP(c, epsilon, True))     # approximate the amount of straight lines between points (epsilon changes amount of lines on a curved edge)
        [x,y,w,h] = cv2.boundingRect(c) # finds origin, max width and max height of each contour
        box_coord_list.append([x,y,w,h])    # adds coordinates to a list
        box_list.append(w*h)    # addss area of ounding rectangle to a list

    max_index = 0
    try:    # error handling
        max_ = box_list[0]  # sets max_ to first element in box_list
        
        for z, item in enumerate(box_list): # itterated through the area values for each contour to find the largest
            if item > max_:
                max_ = item
                max_index = z
        
        [x,y,w,h] = box_coord_list[max_index]   # resets coordinates to only the largest bounding rectange
        max_contour = contours[max_index]   # takes the information from the largest contour out of the list
        
        max_approx_contour = cv2.approxPolyDP(max_contour, epsilon, True) # straightens contours so there are not curved edges
        
        
        # cv2.drawContours(imgc, max_approx_contour,-1,(0,255,0),1)         # paint contours over copy of image
        
        # cv2.rectangle(imgc,  (x-5,y-5), (x+w+5,y+h+5), (255, 0, 0), 2)  # paints bounding rectangle over copy of image
        
        list_of_shapes.append (((x,y), (x+w,y+h))) # adds tuple containing 2 coordinates of coreners of the bounding rectangle
        
        cropped_img = img[y:y+h,x:x+w]  # crops image to bounding recangle
        
        if inp_image == True:   # only executed when taking images from camera as would take too much extra processing to run for each training image
            rgb_colour = find_colour(cropped_img, [x,y,w,h])    
            colour_name = rgb_hsv(rgb_colour)
        
        else:
            colour_name = None      # prevents error when returning values for training images
        
        ellipse = cv2.fitEllipse(max_contour)    # fit an ellipse on the contour
        (center, axes, orientation) = ellipse   # extract the main parameter
        majoraxis_length = max(axes)
        minoraxis_length = min(axes)
        
        # hull = cv2.convexHull(max_approx_contour)
        # how_convex = abs(hull [0][0][0]/hull[0][0][1])
        
        feature1 = cv2.contourArea(max_approx_contour)/cv2.arcLength(max_approx_contour, True)    # divides area by perimeter
        
        feature2 = cv2.isContourConvex(max_approx_contour)  # checks if shape has concavities
        
        feature3 = vertex_approx # approximate the amount of straight lines between points (epsilon changes amount of lines on a curved edge)
        
        feature4 = majoraxis_length/minoraxis_length # find the major over minor axis value (height vs width)
        
        # feature5 = how_convex
        # feature6 = parallel lines
        # feature7 = lines of similar 
        
        
        feature_list = [feature1, int(feature2), feature3, feature4] # puts features in a list
        return feature_list, [x,y,w,h], colour_name, False # returns useful variables
    
    except IndexError:
        return [0,0,0,0], [200,200,0,0], "No Shape Detected", True #returned values to prevent errors in other functions, True signifies error has occured
    except Exception:
        return [0,0,0,0], [200,200,0,0], "No Shape Detected", True #returned values to prevent errors in other functions, True signifies error has occured

 # Function for processing images from camera stream
def input_image(inp_img):
    input_image_features = []
    shapes_temp, bound_rect, colour_name, error_while_checking = find_shapes(inp_img, True)
    input_image_features.append(shapes_temp) 
    
    return input_image_features, bound_rect, colour_name, error_while_checking

 # Function for drawing name of shape or colour over image
def display_text(colour, prediction, bound_rect, transform):
    [x,y,w,h] = bound_rect # unpacks coordinates
    x_tf, y_tf = transform # unpacks the transform x and y values, this allows ofr text to put in differnt places
    cv2.rectangle(img,  (x,y), (x+w,y+h), (255, 0, 0), 2)   # paints rectange over displayed image
    middle_xy = (int(x+(w/2)), int(y+(h/2)))    # finds the centre coordinate of bounding rectangle
    if colour == "Red" or colour == "Orange":
        text_col = (255, 0, 0)
    else:
        text_col = (0, 0, 255)
    cv2.putText(img, prediction, (middle_xy[0]+x_tf, middle_xy[1]+y_tf), cv2.FONT_HERSHEY_DUPLEX, 1, text_col)
   

 # Colorsys wasn't working properly so this is snipet taken from the source code  
def rgb_to_hsv(r, g, b):    
    try:
        r, g, b = r/255.0, g/255.0, b/255.0
        mx = max(r, g, b)
        mn = min(r, g, b)
        df = mx-mn
        if mx == mn:
            h = 0
        elif mx == r:
            h = (60 * ((g-b)/df) + 360) % 360
        elif mx == g:
            h = (60 * ((b-r)/df) + 120) % 360
        elif mx == b:
            h = (60 * ((r-g)/df) + 240) % 360
        if mx == 0:
            s = 0
        else:
            s = (df/mx)*100
        v = mx*100
        return h, s, v
    except TypeError:
        return 150,0,0

 # Function to take average colour in area containing the shape
def find_colour(cropped_img, bound_rect):
    total_r = 0
    total_g = 0
    total_b = 0
    
    image = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # black = np.array([0,0,125])
    # shape = np.array([255,100,255])
    # mask = cv2.inRange(hsv, black, shape)
    # mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) # coverts black and white image back to a 3 channel image to allow merging with colour image
    # mask_colour = cv2.bitwise_and(cropped_img, mask3)   # merges mask and colour image to leave a black background around the shape

    # cv2.imshow("mask", mask_colour)
    # cv2.waitKey()
    ###
    
    
    # reshape = mask_colour.reshape((image.shape[0] * image.shape[1], 3))
    reshape = image.reshape((image.shape[0] * image.shape[1], 3))
    for a in range (0, len(reshape), 1):
        # if reshape[a][0] != 0 and reshape[a][1] != 0 and reshape[a][2] != 0:
        total_r += reshape[a][0]
        total_g += reshape[a][1]
        total_b += reshape[a][2]
    return (total_r/len(reshape), total_g/len(reshape), total_b/len(reshape))
        

 # Function to determine the name of the colour from the average hue value
def rgb_hsv(rgb_value):
        r_colour, g_colour, b_colour = rgb_value
        hsv_colour_temp, _, _ = rgb_to_hsv(r_colour, g_colour, b_colour)
        if 0 <= hsv_colour_temp < 55:
            colour_name =("Orange")
        elif 55 <= hsv_colour_temp < 130:
            colour_name = ("Yellow")
        elif 130 <= hsv_colour_temp < 170:
            colour_name = ("Green")
        elif 170 <= hsv_colour_temp < 200:
            colour_name = ("Cyan")
        elif 200 <= hsv_colour_temp < 235:
            colour_name = ("Blue")
        elif 235 <= hsv_colour_temp < 260:
            colour_name = ("Purple")
        elif 260 <= hsv_colour_temp < 335:
            colour_name = ("Pink")
        elif 335 <= hsv_colour_temp < 360:
            colour_name = ("Red")    
        return colour_name
    

 # Itterates through all training images and finds the features for each
for a in shapes:
    shape = a
    for b in range(20):
        img = cv2.imread("./images2/"+shape+str(b)+".png")
        shape_temp, _, _, _ = find_shapes(img, False)
        features.append(shape_temp)

 # This loop randomises which images are taken for each shape, ensuring there is an even quantity of each shape 
for i in range (6): # 6 indicates the amount of unique shapes
    for b in range (15): # 15 indicates that 15 images for each shape is used for training
        if i == 0:    
            r = random.randint(0, (len(features))/6)
        else:    
            r = random.randint(((20*i)-(len(features))/6), (20*i)-1)
        if r not in train_list: # prevents duplicates of images
            features_shuffled_train_list.append(features[r])
            target_shuffled_train_list.append(shape_class_list[r])
            train_list.append(r)
        else:
            b-=1

 # Creates list containing only testing features / names
for i in range (len(features)):
    if i not in train_list:
        features_shuffled_test_list.append(features[i])
        target_shuffled_test_list.append(shape_class_list[i])
        
 # Creates clasifier for machine learning
mlp = MLPClassifier(hidden_layer_sizes=(350,350), activation='relu', solver='adam', alpha=0.0000001, learning_rate='adaptive', beta_1=0.9, beta_2=0.999, epsilon=1e-5, max_iter=10000) #hidden_layer_sizes=(35,), activation='tanh', solver='adam', alpha=0.000001, learning_rate='adaptive', beta_1=00.9, beta_2=0.999, epsilon=1e-8, max_iter=10000

 # Trains classifier with training images
fitted = mlp.fit(features_shuffled_train_list, target_shuffled_train_list)


print(round(fitted.score(features_shuffled_test_list , target_shuffled_test_list)*100,2),"%")

print("Would you like to use a camera or a saved image?")
valid_data = False
while valid_data == False:
    cam_or_img = input ("cam or img:  ")
    if cam_or_img in ("cam", "img"):
        valid_data = True
    else: 
        print ("Invalid Input!")
        
while cam.isOpened():
    if cam_or_img == "img":
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        img = cv2.imread(file_path)

    else:
        exp = cv2.getTrackbarPos("Exposure", "Trackbars") # reads position from trackbar
        cam.set(cv2.CAP_PROP_EXPOSURE,-exp) # sets camera exposure
        _, img = cam.read()   # takes frame from camera stream
        

    
    inp_features, bound_rect, colour_name, error_while_checking = input_image(img)
    if error_while_checking == False:   # only runs if there were no errors while finding features to prevent crashing
        predict_shape = fitted.predict(np.array(inp_features))[0] # fits features from current frame to classifier and outputs a list of predictions (decending probobility)
        display_text(colour_name, predict_shape, bound_rect, (-25,20))
    
    display_text(colour_name, colour_name, bound_rect, (-25,-20))
    
     # Display current frame to user
    cv2.namedWindow('picture', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow('picture',img)

    if cam_or_img == "img":
        key = cv2.waitKey(0) # waits indefinately for keyboard input
        break
    # exit command using "ESC" as trigger
    key = cv2.waitKey(1) # waits 1ms for keyboard input
    if key == 27:
        break

 # Closing tag for open cv2 windows to prevent hanging
cv2.destroyAllWindows()



        

