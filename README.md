# cs153_finalProject

This code allows for the synchronization of echocardiograms. To run it, use the main() function. This function takes in one required argument and two optional arguments. The required argument is a list of filenames. The filenames must provide the path to the dicom files being analyzed. The first optional argument is style, which denotes the layout of the image. The only supported style is 0, which represents the straight view with an electrocardiogram along the bottom. The final optional argument is intermediateSteps, which defaults to False. When set to True, the program will save the original videos, electrocardiogram annotated with the tracking, graph of the electrocardiogram, and graph of the critical points in addition to the final synchronized videos.

Example call to the function:
main(['filepath1.dcm', 'filepath2.dcm'], style = 0, intermediateSteps = True)

This program does require the packages:

matplotlib.pyplot

cv2

numpy

pydicom

os

scipy.signal

