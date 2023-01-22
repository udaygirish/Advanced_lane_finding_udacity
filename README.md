## Advanced Lane Finding Project
---
####Introduction
This project is a part of coursework of Udacity Self Driving Nanodegree. The projects aim is to revisit the learnings from Image processing and Computer vision concepts and apply them to solve a real world simple version of a complex problem - Lane Identification for Self Driving Cars. Here we mostly use Traditional methods, most of the outputs are based on Thresholding methods and some Poly Fit math, I have also attached a sample video of State of the Art Latest deep learning approaches to just show the progress we made as a Self Driving community. Code for DL approaches not available here, This repo is only for Traditional methods, for DL approaches I have provided references.



####Goal 
To find the Lane boundaries, Curvature and Vehicles position from a Car Frontal Cam View.

####Steps
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Sample_Output
![Sample_Output](./output_images/test6_output.jpg)


####Pipeline - Flow Diagram
![Pipeline Block Diagram](./readme_images/adv_lane_finding_pipeline.png) 


####Code-Folder Explanation
![Folder_Structure](./readme_images/Folder_Structure.jpg)

* lib is the source folder for all the modules.
* img_operations.py contains all the objects and methods necessary for image transformation
* cam_calibration.py contains all the objects and methods necessary for Camera Calibration and calculation of Distortion Coefficient.
* lane_finder.py contains all the objects and methods necessary for Lane Detection, Identification, Curvature measurement and Helper functions for plotting the output on Images or videos
* main.py is the Main program to run the Code. It imports all the functions above and capable to work with both video and image.
* test_images contains test images
* camera_cal contains the calibration Images
* output_images contains the outputs of test images


####How to Run
The code can be executed with the help of simple python3 main.py call with arguments
*****Arguments*****

        * -i/--input ==> To give the Input Path
        * -it/--input_type ==> To specify input type <video/image>
        * -o/--output_path ==> Output Path, default set to output_images -> Folder path only (Default Output File is saved with <input_file_name>_output.*)

*****Run Command*****

        python3 main.py -i <video_path/image_path> --it <video/image> 
        -o <output_path>

        Example Command:
        python3 main.py -i ./project_video.mp4 -it video -o ./output_images/

####Camera Calibration

When a camera looks at a 3D Object and tries to register it to a 2D Image. The 2D representation becomes a bit distorted because of the 3D and angular aspects of the Object. Usually this involves multiple aspects such as the Position of the object, Angle w.r.t plane /camera plane, Camera Relative position, Camera Focal length etc. These distortion coefficients and calibration are calculated with the help of a Pinhole Model approach.
This Calibration of Camera is available as a module in OpenCV called 
cv2.calibrate

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

#### Pipeline (Single Image Based) - Explanation

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

####Pipeline (video) Links

***Traditional Methods***

1. [Project Video Output](https://drive.google.com/file/d/1r1HQK5NaE-fCgoD5vuhdZX-uQVtEaA_O/view?usp=sharing)
2. [Challenge Video Output](https://drive.google.com/file/d/1Da1R5fr3ajiQUE_16a651buRiW1iWeYN/view?usp=sharing)
3. [Harder Challenge Video Output](https://drive.google.com/file/d/1LXrBjy6OwV7ngiTOmSeSIThl1awAhL1P/view?usp=sharing)

***DeepLearning Methods***
This is just to show some current SOTA methods I have gone through and tried to experiment with to understand how we are solving this problem now.

1.[Project Video DL Output]
2.[Challenge Video DL Output]
3.[Harder Challenge Video Output]


### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. 


#### References

