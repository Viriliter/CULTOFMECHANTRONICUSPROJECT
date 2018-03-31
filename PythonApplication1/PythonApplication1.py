import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
from collections import deque
from line import Line
'''fkdsfsd'''
def fpsCounter(vid):
    fps = vid.get(cv2.CAP_PROP_FPS)
    print("Fps: {0}".format(fps))

def CreateMask(frame):
    #Create white line mask
    lowerW = np.uint8([230, 230, 230])
    upperW = np.uint8([255, 255, 255])
    whiteMask = cv2.inRange(frame,lowerW,upperW)
    #Create yellow line mask
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowerY = np.uint8([50, 50, 50])
    upperY = np.uint8([110, 255, 255])
    yellowMask = cv2.inRange(frame,lowerY,upperY)
    #Fuse white and yellow masks
    mask = cv2.bitwise_or(whiteMask,yellowMask)
    return cv2.bitwise_and(frame,frame,mask = mask)

def ConvertToSChannel(frame):
    HLS = cv2.cvtColor(frame,cv2.COLOR_BGR2HLS)
    h,l,s = cv2.split(HLS)
    return HLS

def Sobel(frame):
    sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1)
    scaled_sobel = np.uint8(100*abs_sobelx/np.max(abs_sobelx))
    return scaled_sobel

def BinaryThreshold(frame, thresh_min, thresh_max):
    xbinary = np.zeros_like(frame)
    xbinary[(frame >= thresh_min) & (frame <= thresh_max)] = 255
    return xbinary

def AbsSobelThresh(frame, orient='x', sobel_kernel=3, thresh_min=0, thresh_max=255):
    if orient == 'x':
        yorder = 0
        xorder = 1
    else:
        yorder = 1
        xorder = 0
    sobel = cv2.Sobel(frame, cv2.CV_64F, xorder, yorder)
    abs_sobel = np.absolute(sobel)
    scaled = np.uint8(255.0*abs_sobel/np.max(abs_sobel))
    return BinaryThreshold(scaled,thresh_min,thresh_max)

def MagThresh(img, sobel_kernel=3, thresh_min=0, thresh_max=255):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    return BinaryThreshold(gradmag, thresh_min, thresh_max)

def CombineGradients(frame):
    sChannel = ConvertToSChannel(frame)
    '''sobelX = AbsSobelThresh(sChannel, thresh_min=200, thresh_max=255)
    sobelY = AbsSobelThresh(sChannel, orient='y', thresh_min=200, thresh_max=255)'''
    sobelX = MagThresh(sChannel, thresh_min=200, thresh_max=255)
    sobelY = MagThresh(sChannel, thresh_min=200, thresh_max=255)
    combined = np.zeros_like(sobelX)
    combined[((sobelX == 255) & (sobelY == 255))] = 255
    return combined


def ConvertToGray(frame):
    '''
    (thresh, im_bw) = cv2.threshold(frame, 220, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = 127
    im_bw = cv2.threshold(frame, thresh, 255, cv2.THRESH_BINARY)[1]
    '''
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Create binary based on detected pixels
    binary_threshold = np.zeros_like(gray)
    #binary_threshold[(gray > 0)] = 1
    return binary_threshold

def CreateTrapzoid(frame, BottomWidth,UpperWidth,Height,Xbias=0,Ybias=0):
    frame_size = (frame.shape[1],frame.shape[0])
    src = np.array([[frame_size[0]/2-UpperWidth/2+Xbias,frame_size[1]-Height+Ybias],[frame_size[0]/2+UpperWidth/2+Xbias,frame_size[1]-Height+Ybias],[frame_size[0]/2+BottomWidth/2+Xbias,frame_size[1]+Ybias],[frame_size[0]/2-BottomWidth/2+Xbias,frame_size[1]+Ybias]],np.float32)
    #dst = np.array([[frame_size[0]/2-BottomWidth/2,0], [frame_size[0]/2+BottomWidth/2,0], [frame_size[0]/2+BottomWidth/2,Height] , [frame_size[0]/2-BottomWidth/2,Height]  ],np.float32)
    if(BottomWidth>UpperWidth):
        maxWidth = BottomWidth
    else:
        maxWidth = UpperWidth
    dst = np.array([[frame_size[0]/2-BottomWidth/2,0], [frame_size[0]/2+BottomWidth/2,0], [frame_size[0]/2+BottomWidth/2,frame_size[1]] , [frame_size[0]/2-BottomWidth/2,frame_size[1]]  ],np.float32)

    return src,dst

def ConvertToBirdView(frame):
    frame_size = (frame.shape[1],frame.shape[0])
    '''src = np.array([[290, 238] ,[349, 238], [498, 337],[192, 337]],np.float32)
    dst = np.array([[192, 0], [498, 0], [498, 360] , [192, 360]  ],np.float32) '''
    #src = np.array([[270, 220] ,[340, 220], [590, 360],[50, 360]],np.float32)
    #dst = np.array([[50, 0], [590, 0], [590, 360] , [50, 360]  ],np.float32) 
    
    src,dst = CreateTrapzoid(frame,570,220,100)
    M = cv2.getPerspectiveTransform(src,dst)
    #M = np.fliplr([M])[0]
    return cv2.warpPerspective(frame, M, frame_size, flags=cv2.INTER_LINEAR)

def ConvertFromBirdView(frame):
    frame_size = (frame.shape[1],frame.shape[0])
    '''src = np.array([[290, 238] ,[349, 238], [498, 337],[192, 337]],np.float32)
    dst = np.array([[192, 0], [498, 0], [498, 360] , [192, 360]  ],np.float32) '''
    #src = np.array([[270, 220] ,[340, 220], [590, 360],[50, 360]],np.float32)
    #dst = np.array([[50, 0], [590, 0], [590, 360] , [50, 360]  ],np.float32) 
    
    src,dst = CreateTrapzoid(frame,570,220,100)
    M = cv2.getPerspectiveTransform(dst,src)
    #M = np.fliplr([M])[0]
    return cv2.warpPerspective(frame, M, frame_size, flags=cv2.INTER_LINEAR)

def BinaryDistribution(frame):
    histogram = np.sum(frame[int(frame.shape[1]/2):,:], axis=0)

    midpoint = np.int(histogram.shape[1]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

def sliding_window(binary_warped):
    out_img = (np.dstack((binary_warped, binary_warped, binary_warped)) * 255).astype(np.uint8)
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        #cv2.rectangle(out_img, (win_xleft_low,win_y_low), (win_xleft_high,win_y_high), color=(0,255,0), thickness=2) # Green
        #cv2.rectangle(out_img, (win_xright_low,win_y_low), (win_xright_high,win_y_high), color=(0,255,0), thickness=2) # Green
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]  
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    #print(left_fit) # to measure tolerances
    print(left_fit)
    # Stash away polynomials
    left_line.current_fit = left_fit
    right_line.current_fit = right_fit
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    out_img[ploty.astype('int'),left_fitx.astype('int')] = [0, 255, 255]
    out_img[ploty.astype('int'),right_fitx.astype('int')] = [0, 255, 255]

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, deg=2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, deg=2)

    # Calculate radii of curvature in meters
    y_eval = np.max(ploty)  # Where radius of curvature is measured
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # Stash away the curvatures  
    left_line.radius_of_curvature = left_curverad  
    right_line.radius_of_curvature = right_curverad
    
    return left_fit, right_fit, left_curverad, right_curverad, out_img

def non_sliding(binary_warped, left_fit, right_fit):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin))
        & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin))
        & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2) 
    except:
        return left_line.current_fit, right_line.current_fit, left_line.radius_of_curvature, right_line.radius_of_curvature, None
    
    else:
        # Check difference in fit coefficients between last and new fits  
        left_line.diffs = left_line.current_fit - left_fit
        right_line.diffs = right_line.current_fit - right_fit
        if (left_line.diffs[0]>0.001 or left_line.diffs[1]>0.4 or left_line.diffs[2]>150):
            return left_line.current_fit, right_line.current_fit, left_line.radius_of_curvature, right_line.radius_of_curvature, None
        #print(left_line.diffs)
        if (right_line.diffs[0]>0.001 or right_line.diffs[1]>0.4 or right_line.diffs[2]>150):
            return left_line.current_fit, right_line.current_fit, left_line.radius_of_curvature, right_line.radius_of_curvature, None
        #print(right_line.diffs)
        
        # Stash away polynomials
        left_line.current_fit = left_fit
        right_line.current_fit = right_fit

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, deg=2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, deg=2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Calculate radii of curvature in meters
        y_eval = np.max(ploty)  # Where radius of curvature is measured
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])     

        # Stash away the curvatures  
        left_line.radius_of_curvature = left_curverad  
        right_line.radius_of_curvature = right_curverad

        return left_fit, right_fit, left_curverad, right_curverad, None
    
def draw_lane(undistorted, binary_warped, left_fit, right_fit, left_curverad, right_curverad):
    
    # Create an image to draw the lines on
    warped_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warped = np.dstack((warped_zero, warped_zero, warped_zero))    
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]   
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    midpoint = np.int(undistorted.shape[1]/2)
    middle_of_lane = (right_fitx[-1] - left_fitx[-1]) / 2.0 + left_fitx[-1]
    offset = (midpoint - middle_of_lane) * xm_per_pix

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warped, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    img_size = (undistorted.shape[1], undistorted.shape[0])
    unwarped = ConvertFromBirdView(color_warped)
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted, 1, unwarped, 0.3, 0)
    radius = np.mean([left_curverad, right_curverad])

    # Add radius and offset calculations to top of video
    cv2.putText(result,"L. Lane Radius: " + "{:0.2f}".format(left_curverad/1000) + 'km', org=(50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1, color=(255,255,255), lineType = cv2.LINE_AA, thickness=2)
    cv2.putText(result,"R. Lane Radius: " + "{:0.2f}".format(right_curverad/1000) + 'km', org=(50,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1, color=(255,255,255), lineType = cv2.LINE_AA, thickness=2)
    cv2.putText(result,"C. Position: " + "{:0.2f}".format(offset) + 'm', org=(50,150), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1, color=(255,255,255), lineType = cv2.LINE_AA, thickness=2)

    return result

def ShowLane(frame,undistorted,nbins=10):
    bins = nbins
    l_params = deque(maxlen=bins)
    r_params = deque(maxlen=bins)
    l_radius = deque(maxlen=bins)
    r_radius = deque(maxlen=bins)
    weights = np.arange(1,bins+1)/bins

    if len(l_params)==0:
        left_fit, right_fit, left_curverad, right_curverad, _ = sliding_window(frame)
    else:
        left_fit, right_fit, left_curverad, right_curverad, _ = non_sliding(frame,
                                                                np.average(l_params,0,weights[-len(l_params):]),
                                                                np.average(r_params,0,weights[-len(l_params):]))
        
    l_params.append(left_fit)
    r_params.append(right_fit)
    l_radius.append(left_curverad)
    r_radius.append(right_curverad)
    annotated_frame = draw_lane(undistorted,
                                frame,
                                np.average(l_params,0,weights[-len(l_params):]),
                                np.average(r_params,0,weights[-len(l_params):]),
                                np.average(l_radius,0,weights[-len(l_params):]),
                                np.average(r_radius,0,weights[-len(l_params):]))
    return annotated_frame

'''
def FindLines(frame, nwindows=9, margin=110, minpix=50):

    histogram = np.sum(frame[frame.shape[1]//2:,:], axis=0)
    out_frame = np.dstack((frame, frame, frame))*255

    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = np.int(frame.shape[0]/nwindows)
    nonzero = frame.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = frame.shape[0] - (window+1)*window_height
        win_y_high = frame.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

    cv2.rectangle(out_frame,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
    cv2.rectangle(out_frame,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)

    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
    if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    left_fit_m = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_m = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    return (left_fit, right_fit, left_fit_m, right_fit_m, left_lane_inds, right_lane_inds, out_frame, nonzerox, nonzeroy)
    
def VisualizeLanes(frame):
    
    left_fit, right_fit, left_fit_m, right_fit_m, left_lane_inds, right_lane_inds, out_frame, nonzerox, nonzeroy = FindLines(frame)
    
    ploty = np.linspace(0, frame.shape[1]-1, frame.shape[1] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_frame[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_frame[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    ax.imshow(out_frame)
    ax.plot(left_fitx, ploty, color='yellow')
    ax.plot(right_fitx, ploty, color='yellow')
    return ( left_fit, right_fit, left_fit_m, right_fit_m )

def ShowLaneOnFrame(frame, cols = 2, rows = 3, figsize=(15,13)):
    imageAndFit = []
    imgLength = len(images)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    indexes = range(cols * rows)
    imageAndFit = []
    for ax, index in zip(axes.flat, indexes):
        if index < imgLength:
            imagePathName, image = images[index]
        left_fit, right_fit, left_fit_m, right_fit_m = VisualizeLanes(image, ax)
    ax.set_title(imagePathName)
    ax.axis('off')
    left_fit, right_fit, left_fit_m, right_fit_m = VisualizeLanes(frame)
    imageAndFit.append( ( frame, left_fit, right_fit, left_fit_m, right_fit_m ) )
    return imageAndFit
'''

def RemoveNoise(frame):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(frame, kernel, iterations = 1)

def ChangeBrightness(frame, intense=25.0):
    array_beta = np.array([25.0])
    cv2.add(frame, array_beta, frame)
    return frame

def ChangeContrast(frame, intense=1.5):
    array_alpha = np.array([intense])
    cv2.multiply(frame, array_alpha, frame)  
    return frame

def ApplyBlurring(frame, kernel_size=15):
    """
    kernel_size must be postivie and odd
    """
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

def HistogramCalculator(frame):
    hist_full = cv2.calcHist([frame],[0],None,[256],[0,256])
    plt.plot(hist_full)
    plt.show()
    '''frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    sumArray = np.zeros(frameWidth)
    for i in range(0,frameWidth-1):
        for j in range(0,frameHeight-1):
            sum = 0
            sum += frame[j,i]
        sumArray[i] = sum
    plt.plot(sumArray)
    plt.show()'''

def testVideo():
    vid = cv2.VideoCapture('videoplayback.mp4')
    while(vid.isOpened()):
        ret, frame = vid.read()
        #src,dst = CreateTrapzoid(frame,570,220,100)
        cv2.imshow('UnMasked',frame)
        #r = cv2.selectROI(frame)
        #print(r)
        undistorted = frame
        #frame = cv2.fillPoly(frame,np.int_([src]),(0,255,255))

        #frame = ChangeContrast(frame)
        #frame = CreateMask(frame)
        #frame = ConvertToSChannel(frame)


        #frame = Sobel(frame)
        #frame = BinaryThreshold(frame,200,255)
        #frame = CombineGradients(frame)
        frame = cv2.Canny(frame,200,255)
        frame = ConvertToBirdView(frame)
        cv2.imshow('Canny',frame)
        frame = ShowLane(frame,undistorted)
        #frame = ShowLaneOnFrame(frame)
        #frame = RemoveNoise(frame)
        #frame = ChangeContrast(frame)
        #frame = ChangeBrightness(frame)


        cv2.imshow('Masked',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

left_line = Line()
right_line = Line()
testVideo()
