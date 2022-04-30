# Load in the important dictionaries
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pydicom
import os
import scipy.signal as sig


def main(filenames, style = 0, intermediateSteps = False):
    """
    This is the main function that takes in the path to 
    the folder with the dicom files. It expects that all
    videos are held within the same folder. Alternatively,
    it can take in a list of the paths to the files. 
    
    style is a parameter that represents the format of the 
    files. 0 is the blank echo with electrocardiogram on the
    bottom. No other formats are supported at this time.

    intermediateSteps is a boolean that represents whether or
    not the program will write the intermediate steps out to 
    a file.
    """

    # Remove / from filenames to use as labels
    names = []
    for file in filenames:
        names += [file.replace("/", "_")[1:]]

    # Read each file and place the dicom object in a list
    dicomObjects = []

    for file in filenames:
        ds = pydicom.read_file(file)
        dicomObjects += [ds]
    
    # Plot the electrocardiogram signal for every file
    #       Returns the x and y plots, plus a marking of 
    #       which belong to frames
    x_plots = []
    y_plots = []
    frameTracking = []
    toremove = []
    for j in range(len(dicomObjects)):
        ds = dicomObjects[j]
        print("Now plotting", names[j])
        result = electroCardiogramPlot(ds, style, intermediateSteps, names[j])
        if result != 0:
            x_plots += [result[0]]
            y_plots += [result[1]]
            frameTracking += [result[2]]
        else:
            toremove += [j]
    
    for j in range(len(toremove), -1):
        dicomObjects.remove(dicomObjects[j])
        filenames.remove(filenames[j])
        names.remove(names[j])
    
    # Find the expected number of beats for each video
    num_beats = []
    for j in range(len(dicomObjects)):
        heart_rate = ds.HeartRate
        frame_len = ds.FrameTime
        frame = ds.NumberOfFrames
        total_len = ((frame_len / 1000) * frame) / 60
        beats = int(heart_rate * total_len)
        num_beats += [beats]


    # Find the critical points from each graph
    print("Beginning critical points analysis")
    criticalPoints = []
    remove = []
    min_beats = min(num_beats) - 1

    if min_beats == 0:
        print("Cannot align videos: not a full cardiac cycle")

    for j in range(len(dicomObjects)):
        critP = findCriticalPoints(x_plots[j], y_plots[j], frameTracking[j], num_beats[j], intermediateSteps, names[j], min_beats)
        if critP != 0:
            criticalPoints += [critP]
        else:
            print("Could not locate start of heartbeats for video", names[j], "removing from set")
            remove += [j]
        
    # Remove any videos that return an error
    for j in range(len(remove), -1):
        dicomObjects.remove(dicomObjects[j])
        filenames.remove(filenames[j])
        names.remove(names[j])
        x_plots.remove(x_plots[j])
        y_plots.remove(y_plots[j])
        frameTracking.remove(frameTracking[j])
        num_beats.remove(num_beats[j])
        criticalPoints.remove(criticalPoints[j])

    # Determine which sequence will be the reference sequence
    # Use first the number of critical points, then the number of frames contained
    ref_ranking = []
    for j in range(len(dicomObjects)):
        num = len(criticalPoints[j])
        frameRange = criticalPoints[j][-1][0] - criticalPoints[j][0][0]
        ref_ranking += [[num, frameRange, criticalPoints[j], j]]
    ref_ranking.sort(reverse = True)
    referenceSequence = ref_ranking[0][2]
    reference_number = ref_ranking[0][3]
    print("Reference video is", names[reference_number])


    # Synchronize every other sequence with the reference
    print("Synchronizing frames")
    seq_of_frames = []
    for j in range(len(dicomObjects)):
        new_seq = synchronizeFrames(referenceSequence, criticalPoints[j])
        seq_of_frames += [new_seq]

    # Output the video from each sequence
    print("Writing output videos")
    for j in range(len(dicomObjects)):
        image_data = dicomObjects[j].pixel_array
        num_frames, rows, cols, color = image_data.shape
        ArrayDicom = np.zeros((num_frames, rows, cols, color), dtype=dicomObjects[j].pixel_array.dtype)
        ArrayDicom[:,:,:] = dicomObjects[j].pixel_array

        size = (cols, rows)
        full = cv2.VideoWriter('./synchronizedVideos_' + names[j] + '.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

        for n in seq_of_frames[j]:
            full.write(ArrayDicom[n])
        full.release()

    print("Testing results")
    # Test the synchronization of the electrocardiogram
    electro_measure = []
    electro_baseline = []

    # Prep the reference information
    reference_array_original = dicomObjects[reference_number].pixel_array
    reference_original = electrocardiogramSynchronizationHelper(reference_array_original, style, len(reference_array_original))

    reference_Array = []
    reference_vidcap = cv2.VideoCapture('./synchronizedVideos_' + names[reference_number] + '.avi')
    reference_success,reference_image = reference_vidcap.read()
    while reference_success:
        reference_Array += [reference_image]
        reference_success,reference_image = reference_vidcap.read()
    reference_new = electrocardiogramSynchronizationHelper(reference_Array, style, len(reference_Array))

    # Run across every other video
    for j in range(len(dicomObjects)):
        print("Testing ", names[j])
        measure = electrocardiogramSynchronization('./synchronizedVideos_' + names[j] + '.avi', reference_new, style, original = False)
        baseline = electrocardiogramSynchronization(filenames[j], reference_original, style, original = True)
        electro_measure += [measure]
        electro_baseline += [baseline]


    print("Testing pixel differences")
    # Test the synchronization of the videos
    pixel_measure = []
    pixel_baseline = []

    for j in range(len(dicomObjects)):
        measure = pixelSynchronization('./synchronizedVideos_' + names[j] + '.avi', './synchronizedVideos_' + names[reference_number] + '.avi', original = False)
        baseline = pixelSynchronization(filenames[j], filenames[reference_number], original = True)
        pixel_measure += [measure]
        pixel_baseline += [baseline]

    # Report results
    print(len(dicomObjects), "files were successfully processed")
    print("Average starting electrocardiogram difference was", sum(electro_baseline)/len(electro_baseline))
    print("Average final electrocardiogram difference was", sum(electro_measure)/len(electro_measure))

    print("Average starting mean squared pixel difference was", sum(pixel_baseline)/len(pixel_baseline))
    print("Average final mean squared pixel difference was", sum(pixel_measure)/len(pixel_measure))

    return


def electroCardiogramPlot(ds, style, intermediateSteps, name):
    """
    This function takes in the dicom object, the style it is in
    (for use to isolate the electrocardiogram), whether or not
    to save the intermediate steps as files, the name of the
    file the dicom object came from, and whether or not to interpolate
    between frames.
    By reading the dicom object, it isolates the electrocardiogram 
    and plots its movement across frames. It returns the x coordinates,
    the y coordinates, and a list that stores information on whether or
    not the x coordinate is directly related to a frame
    """
    # Pull relevant data from the file
    image_data = ds.pixel_array
    num_frames, rows, cols, color = image_data.shape
    ArrayDicom = np.zeros((num_frames, rows, cols, color), dtype=ds.pixel_array.dtype)
    ArrayDicom[:,:,:] = ds.pixel_array

    # Initialize return values
    xyfunc = []
    frameNum = []

    # If intermediateSteps is true, save the original video here
    if intermediateSteps:
        size = (cols, rows)
        full = cv2.VideoWriter("./originalVideos_" + name +".avi",cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

        for i in range(num_frames):
            full.write(ArrayDicom[i])
        cv2.destroyAllWindows()
        full.release()

    # Step through every frame
    for i in range(num_frames):
        img = ArrayDicom[i]
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        h, w, s = hsv_img.shape

        # Set our upper and lower threshold for green screen
        upper = (150, 255, 255)
        lower = (20, 85, 90)

        # In the expected case
        if style == 0:
            # Blank out the top two thirds part
            for row in range(w):
                for col in range(int((2*h)/3)):
                    hsv_img[col][row] = [0,0,0]
        else:
            print("This style is implemented yet")

        # Initialize the mask
        mask = hsv_img.copy()

        # Loop through initial image and check each pixel against the threshold
        mask = cv2.inRange(hsv_img, lower, upper)

        # make a black and white mask
        new_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        heights = []
        coords = []

        # Find the coords of all white pixels and keep track of 
        #       the y coord separately
        for l in range(w):
            for j in range(h):
                if mask[j][l] == 255:
                    heights += [j]
                    coords += [[j, l]]

        # Find the bar and sanity check the placement
        while True:
            line = min(heights)
            tip = []
            nextLine = []
            for m in coords:

                # Find all pixels at bar tip
                if m[0] == line:
                    tip += [m]

                    # Check the pixels directly beneath them
                    if mask[m[0] + 1][m[1]] == 255 and mask[m[0] + 2][m[1]] == 255:
                        nextLine += [m[1]]

            # Ensure the bar is at least two pixels wide and extends down
            if len(tip) > 2 and len(nextLine) > 2:
                break
            
            # Highest pixel failed, print error message and try again
            for m in tip:
                heights.remove(m[0])
            
            # If it has gone through every possiblity, print error and return
            if len(heights) == 0:
                print("Was not able to locate the electrocardiogram bar at frame", i, "in video", name, ": removing from file set")
                return 0

            print("Failed attempt to locate electrocardiogram bar at frame", i, "in video", name, "trying again")

        
        # Find the starting x-coordinate we are looking for
        leftMost = w
        for c in tip:
            leftMost = min(leftMost, c[1])

        # Sanity check the data point
        maxHeights = max(heights)
        minHeights = min(heights)

        # Should be in middle of bar
        acceptableRange = [minHeights, maxHeights]

        # Track the location of the electrocardiogram since the last frame
        while True:

            # We look immediately to the left of the bar
            datapoint = leftMost - 1

             # For the first frame, set lastCol to be one pixel before
            if i == 0:
                lastCol = datapoint - 1

            # Case where no data point was found
            if leftMost == 0 or leftMost == lastCol:
                print("Could not locate the y data for frame", i, "in video", name, "checking next one")
                break
        
            # Collect all white pixels at that location
            width = []
            for c in coords:
                if datapoint == c[1]:
                    width += [c[0]]
            if len(width) == 0:
                print("Could not locate the y data for frame", i, "in video", name, "checking next one")
                break

            # Average their value and sanity check it before adding it to the list
            y_coord = int(np.mean(width))
            if y_coord > acceptableRange[0] and y_coord < acceptableRange[1]:
                xyfunc += [[datapoint, y_coord]]
                frameNum += [i]

                # If we have moved more than one column since the last frame, repeat the
                #   process for each colum
                if datapoint > lastCol - 1:
                    for b in range(1, datapoint - lastCol):
                        temp = []
                        for c in coords:
                            if datapoint - c[1] == b:
                                temp += [c[0]]
                        temp_av = int(np.mean(temp))
                        if temp_av > acceptableRange[0] and temp_av < acceptableRange[1]:
                            xyfunc += [[datapoint-b, temp_av]]
                            frameNum += [i] # Not on a frame
                    break

                # If we have cycled back to the left of the frame
                elif lastCol > datapoint:
                    
                    # Add the data between the last frame and the right of the image
                    for b in range(lastCol, cols):
                        temp = []
                        for c in coords:
                            if b == c[1]:
                                temp += [c[0]]
                        if len(temp) > 0:
                            temp_av = int(np.mean(temp))
                            if temp_av > acceptableRange[0] and temp_av < acceptableRange[1]:
                                xyfunc += [[datapoint-b, temp_av]]
                                frameNum += [i] # Not on a frame

                    # Add the data between the left of the image and the current frame
                    for b in range(1, datapoint):
                        temp = []
                        for c in coords:
                            if c[1] == b:
                                temp += [c[0]]
                        if len(temp) > 0:
                            temp_av = int(np.mean(temp))
                            if temp_av > acceptableRange[0] and temp_av < acceptableRange[1]:
                                xyfunc += [[datapoint-b, temp_av]]
                                frameNum += [i] # Not on a frame
                    break
            
            # Did not find data one column over - check the next
            leftMost -= 1

        # Reset the most recent frame
        lastCol = datapoint
    
    # Store the data from xyfunc in two lists
    x_plot = []
    y_plot = []
    x_check = []

    for k in range(len(xyfunc)):
        if xyfunc[k][0] not in x_check:
            x_plot += [k]
            x_check += [xyfunc[k][0]]
            y_plot += [-xyfunc[k][1]] # negative to help with spike location

    # If intermediateSteps is true, create the heartbeat graphs for each
    #       file and mark them over the last frame to check their accuracy
    #       against the actual electrocardiogram
    if intermediateSteps:

        # Plot the graph
        plt.plot(x_plot, y_plot)
        plt.savefig("./graphs" +  name + ".jpg")
        plt.clf()

        # Plot the last frame with marked circles
        for k in range(len(xyfunc)):
            if frameNum[k] == 1: # If its a frame coordinate, plot in red
                cv2.circle(new_mask, (xyfunc[k][0], xyfunc[k][1]), 2, (255,0,0), 1)
        for k in range(len(xyfunc)):
            if frameNum[k] == 0: # If its a nonframe coordinate, plot in blue
                cv2.circle(new_mask, (xyfunc[k][0], xyfunc[k][1]), 2, (0,0,255), 1)
        
        plt.imshow(new_mask) # new_mask is still holding data from last frame
        y = cv2.imwrite("./markedFrame" +  name + ".jpg", new_mask)
        plt.clf()

    return x_plot, y_plot, frameNum



def findCriticalPoints(x_plot, y_plot, frames, num_beats, intermediateSteps, name, min_beats):
    """
    Takes as argument the x and y coordinates for
    the heartbeat graph, as well as the tracker for which
    correspond to frames in the original video.

    Also accepts num_beats, used to isolate how many
    heartbeats to find.

    Returns a list of lists, with the first element being
    the frame number of the critical point, and the second
    a number corresponding to which critical point it is.
    The number is a function of the type and
    which heartbeat it is
    """

    # Find the minimum distance between beats
    dist = len(x_plot)/(num_beats + 1)

    # Find the large peaks
    preliminary_peaks = sig.find_peaks(y_plot, height=(None,None), threshold=0, distance=dist, prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None)

    # # Check to make sure we have the right number -- if not, remove the smallest
    # heights = preliminary_peaks[1]['peak_heights']
    prelim_peaks = list(preliminary_peaks[0])

    peaks = prelim_peaks

    # Check to make sure we have at least the right number--it not, the plot is very flat
    #       because find_peaks will find as many as possible. As it should not be flat, return error
    if len(peaks) < num_beats:
        # print("Could not locate beats on graph of ")
        return 0


    criticalPoints = []
    backupPoints = []

    # Separate into individual beats and analyze each one
    for i in range(num_beats-1):
        
        # Set up range to include both peaks
        current_beat = y_plot[peaks[i]:peaks[i+1] + 1]
        
        # Set up range for finding dips
        neg_current_beat = []
        for y in current_beat:
            neg_current_beat += [-y]
        
        # Find the secondary peak, using dist to ensure we get only one
        peak_current_beat_func = sig.find_peaks(current_beat, height=(None,None), threshold=0, distance=dist, prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None)
        peak_current_beat = list(peak_current_beat_func[0])

        # Find all possible valleys
        valley_current_beat_fun = sig.find_peaks(neg_current_beat, height=(None,None), threshold=0, distance=None, prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None)
        valley_current_beats = list(valley_current_beat_fun[0])
        valley_heights_current_beat = valley_current_beat_fun[1]['peak_heights']

        # Match heights to labels and sort to find the largest
        valleys = []
        for j in range(len(valley_current_beats)):
            valleys += [[valley_heights_current_beat[j], valley_current_beats[j]]]
        valleys.sort(reverse=True)

        # Find the two largest and put them in the correct order
        large_valleys = [[valleys[0][1], valleys[0][0]], [valleys[1][1], valleys[1][0]]]
        large_valleys.sort()


        # Create initial beat measurement
        beatPoints = []
        
        # print("Original points are", [peaks[i], large_valleys[0][1] + peaks[i], peak_current_beat[0] + peaks[i], large_valleys[1][1] + peaks[i]])
        # Make sure they are in order -- if not, remove strays
        beatPoints += [[peaks[i], 1]] # peak is always first
        if large_valleys[0][0] <= peak_current_beat[0]:
            beatPoints += [[large_valleys[0][0] + peaks[i], 2], [peak_current_beat[0] + peaks[i], 3]]
        else:
            beatPoints += [[peak_current_beat[0] + peaks[i], 3]]
        if large_valleys[1][0] >= peak_current_beat[0]:
            beatPoints += [[large_valleys[1][0] + peaks[i], 4]]
        
        if len(beatPoints) == 4:
            criticalPoints += [beatPoints]
            # print("All worked: ", criticalPoints)
        else:
            backupPoints += [beatPoints]
            # print("Not all worked: ", backupPoints)
    
    # If we have enough well-done beats, leave them
    # If we have too many, select the first few
    if len(criticalPoints) > min_beats:
        criticalPoints = criticalPoints[:min_beats]
    
    # If we have too few, add in some of the not well formed and print a warning
    while len(criticalPoints) < min_beats:
        criticalPoints += [backupPoints[0]]
        backupPoints = backupPoints[1:]
        print("WARNING: Critical points not reliable for ", name)
    
    # Order the critical points in a useful way
    finalCriticalPoints = []
    for i in range(len(criticalPoints)):
        for j in range(len(criticalPoints[i])):
            finalCriticalPoints += [[criticalPoints[i][j][0], criticalPoints[i][j][1] + i*4]]

    finalCriticalPoints.sort()

    # Output critical points graph
    if intermediateSteps:
        for p in range(len(finalCriticalPoints)):
            if finalCriticalPoints[p][1] % 4 == 1:
                plt.axvline(x = finalCriticalPoints[p][0], color='red')
            elif finalCriticalPoints[p][1] % 4 == 2:
                plt.axvline(x = finalCriticalPoints[p][0], color='blue')
            elif finalCriticalPoints[p][1] % 4 == 3:
                plt.axvline(x = finalCriticalPoints[p][0], color='green')
            else:
                plt.axvline(x = finalCriticalPoints[p][0], color='orange')
        plt.plot(x_plot, y_plot)
        plt.savefig("./criticalPoints" + name + ".jpg")
        plt.clf()
    
    # Find the closest frame to each x-coord
    for p in range(len(finalCriticalPoints)):
        x = finalCriticalPoints[p][0]
        finalCriticalPoints[p][0] = frames[x]
    
    return finalCriticalPoints


def synchronizeFrames(ref_points, other_points):
    """
    Taking in two sequences, align their critical points
    and then find the frames numbers for other_seq such that
    it's critical points line up with the ref_seq
    """
    # Begin by matching the critical points up
    ref_seq = []
    other_seq = []

    # Add only the critical points they both have
    for i in range(len(ref_points)-1):
        for j in range(len(other_points)):
            if ref_points[i][1] == other_points[j][1]:
                ref_seq += [ref_points[i][0]]
                other_seq += [other_points[j][0]]
    
    # Always add the end
    ref_seq += [ref_points[-1][0]]
    other_seq += [other_points[-1][0]]


    return_seq = []
    # Iterate through the reference sequence
    for i in range(len(ref_seq)-1):
        ref_dist = ref_seq[i+1] - ref_seq[i] # Find distance between reference critical points
        dis_align = other_seq[i+1] - other_seq[i] # Find distance between other critical points
        
        # Find their offset
        offset = ref_dist - dis_align

        # If they are the same, add all frames contained in the section
        if offset == 0:
            return_seq += list(range(other_seq[i], other_seq[i+1]))
        
        # If there are extra frames in the other, delete some
        if offset < 0:
            # Start with the original list
            start_frames = list(range(other_seq[i], other_seq[i+1]))
            
            # Find even interval to delete on
            jump = int(len(start_frames)/(-offset))
            
            # Copy so that indexing works
            new = start_frames.copy()

            # Delete offset number of frames
            for i in range(-offset):
                # At roughly even intervals
                new.remove(start_frames[jump*i])
            
            # Add new frames to the return sequence
            return_seq += new
        
        # If there are too few frames in the other, duplicate some
        if offset > 0:

            # Check that there is an offset in the other sequence
            if other_seq[i] != other_seq[i + 1]:
                start = list(range(other_seq[i], other_seq[i+1]))
                # Find even interval to repeat on
                jump = int((len(start))/offset)
                while True:
                    if jump == 0:
                        start = start + start
                        jump = int(len(start)/offset)
                    else:
                        start = start[:ref_dist]
                        jump = int(len(start)/offset)
                        break

                # Copy so that indexing works
                new = start.copy()
                # Repeat offset number of frames
                for i in range(ref_dist - len(start)):

                    new+= [start[jump*i]]


            # If not, copy same frame offset times
            else:
                new = [other_seq[i]] * offset
            return_seq += new
    
    return_seq.sort()

    return return_seq

def electrocardiogramSynchronizationHelper(file, style, frames):
    """
    Does the reading of the plot for the electrocardiogramSynchronization
    """
    # Initialize return values
    xyfunc = []

    # Step through every frame
    for i in range(frames):
        img = file[i]
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        h, w, s = hsv_img.shape

        # Set our upper and lower threshold for green screen
        upper = (150, 255, 255)
        lower = (20, 85, 90)

        # In the expected case
        if style == 0:
            # Blank out the top two thirds part
            for row in range(w):
                for col in range(int((2*h)/3)):
                    hsv_img[col][row] = [0,0,0]
        else:
            print("Not implemented yet")

        # Initialize the mask
        mask = hsv_img.copy()

        # Loop through initial image and check each pixel against the threshold
        mask = cv2.inRange(hsv_img, lower, upper)

        # make a black and white mask
        new_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        heights = []
        coords = []

        # Find the coords of all white pixels and keep track of 
        #       the y coord separately
        for l in range(w):
            for j in range(h):
                if mask[j][l] == 255:
                    heights += [j]
                    coords += [[j, l]]

        # Find the bar and sanity check the placement
        while True:
            line = min(heights)
            tip = []
            nextLine = []
            for m in coords:

                # Find all pixels at bar tip
                if m[0] == line:
                    tip += [m]

                    # Check the pixels directly beneath them
                    if mask[m[0] + 1][m[1]] == 255 and mask[m[0] + 2][m[1]] == 255:
                        nextLine += [m[1]]

            # Ensure the bar is at least two pixels wide and extends down
            if len(tip) > 2 and len(nextLine) > 2:
                break
            
            # Highest pixel failed, print error message and try again
            for m in tip:
                heights.remove(m[0])
    
        # Find the starting x-coordinate we are looking for
        leftMost = w
        for c in tip:
            leftMost = min(leftMost, c[1])

        # Sanity check the data point
        maxHeights = max(heights)
        minHeights = min(heights)

        # Should be in middle of bar
        acceptableRange = [minHeights, maxHeights]

        # Track the location of the electrocardiogram since the last frame
        while True:

            # We look immediately to the left of the bar
            datapoint = leftMost - 1

            # For the first frame, set lastCol to be one pixel before
            if i == 0:
                lastCol = datapoint - 1

            # Case where no data point was found
            if leftMost == 0 or leftMost == lastCol:
                break

            # Collect all white pixels at that location
            width = []
            for c in coords:
                if datapoint == c[1]:
                    width += [c[0]]

            if len(width) == 0:
                break

            # Average their value and sanity check it before adding it to the list
            y_coord = int(np.mean(width))
            if y_coord > acceptableRange[0] and y_coord < acceptableRange[1]:
                xyfunc += [y_coord]
                break
        
            # Did not find data one column over - check the next
            leftMost -= 1

        # Reset the most recent frame
        lastCol = datapoint
    
    return xyfunc

def electrocardiogramSynchronization(other, reference, style, original = False):
    """
    Reads in the data from other and reference, using file format 
    specified by boolean original (true means a dicom file, false an
    avi file).

    Compares the placement of the electrocardiogram in each frame
    and returns the average difference between the two, using the
    full amount of frames for the shorter video
    """

    # Read in the file data if it is in dicom format
    if original:
        other_ds = pydicom.read_file(other)

        other_image_data = other_ds.pixel_array
        other_num_frames, other_rows, other_cols, other_color = other_image_data.shape
        Other_Array = np.zeros((other_num_frames, other_rows, other_cols, other_color), dtype=other_ds.pixel_array.dtype)
        Other_Array[:,:,:] = other_ds.pixel_array

    # Read in the file data if it is in avi format
    else:
        Other_Array = []
        other_vidcap = cv2.VideoCapture(other)
        other_success,other_image = other_vidcap.read()
        while other_success:
            Other_Array += [other_image]
            other_success,other_image = other_vidcap.read()
        other_num_frames = len(Other_Array)

    comparison = electrocardiogramSynchronizationHelper(Other_Array, style, min(other_num_frames, len(reference)))
    
    # Find the difference between each frame
    difference = []
    for i in range(len(comparison)):
        difference += [abs(comparison[i]-reference[i])]
    
    return sum(difference)/len(difference)
    

def pixelSynchronization(other, reference, original = False):
    """
    Reads in the data from other and reference, using file format 
    specified by boolean original (true means a dicom file, false an
    avi file).

    Compares the pixel difference in each frame using mean squared error
    and returns the average difference between the two, using the
    full amount of frames for the shorter video
    """
    # Read in the file data if it is in dicom format
    if original:
        other_ds = pydicom.read_file(other)
        reference_ds = pydicom.read_file(reference)

        other_image_data = other_ds.pixel_array
        other_num_frames, other_rows, other_cols, other_color = other_image_data.shape
        Other_Array = np.zeros((other_num_frames, other_rows, other_cols, other_color), dtype=other_ds.pixel_array.dtype)
        Other_Array[:,:,:] = other_ds.pixel_array

        reference_image_data = reference_ds.pixel_array
        reference_num_frames, reference_rows, reference_cols, reference_color = reference_image_data.shape
        reference_Array = np.zeros((reference_num_frames, reference_rows, reference_cols, reference_color), dtype=reference_ds.pixel_array.dtype)
        reference_Array[:,:,:] = reference_ds.pixel_array
    
    else:
        Other_Array = []
        other_vidcap = cv2.VideoCapture(other)
        other_success,other_image = other_vidcap.read()
        while other_success:
            Other_Array += [other_image]
            other_success,other_image = other_vidcap.read()
        other_num_frames = len(Other_Array)

        reference_Array = []
        reference_vidcap = cv2.VideoCapture(reference)
        reference_success,reference_image = reference_vidcap.read()
        while reference_success:
            reference_Array += [reference_image]
            reference_success,reference_image = reference_vidcap.read()
        reference_num_frames = len(reference_Array)

    difference = []

    # Check only the first parts of the videos
    num_frames = min(other_num_frames, reference_num_frames)

    for i in range(num_frames):
        diff = mse(Other_Array[i], reference_Array[i])
        difference += [diff]

    
    return sum(difference)/len(difference)

    

# Code taken from https://pyimagesearch.com/2014/09/15/python-compare-two-images/
def mse(imageA, imageB):
	'''
    Calculates mean squared error
    '''
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err