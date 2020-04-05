import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
from moviepy.editor import VideoFileClip

import calibrating_camera as cc
from preproc_images import preproc_images
from line_fit import *

with open('./calibrateCamera.pickle','rb') as caf:
    calibrateCamera = pickle.load(caf)
comeraMatrix = calibrateCamera['comeraMatrix']
distCoeffs = calibrateCamera['distCoeffs']

with open('./M.pickle','rb') as mmf:
    M = pickle.load(mmf)
with open('./Minv.pickle','rb') as mmf:
    Minv = pickle.load(mmf)

windows_cfg = (9,80,100) #(9,80,50)
        
N_LastToAvg = 10
detected = False
slow_fit_again = 0
left_line = Line(N_LastToAvg)
right_line = Line(N_LastToAvg) 

# MoviePy video annotation will call this function 
def line_fit_video(src): 

    global comeraMatrix, distCoeffs, M, Minv, slow_fit_again
    global windows_cfg, curverad_cfg
    global left_line, right_line, detected 
    
    undisted,warped, combined_binary, roi_combined_binary = preproc_images(src, comeraMatrix, distCoeffs, M,)

    h,w = warped.shape  
    histogram_cfg = (h//2,h,0,w)
    curverad_cfg = (h-1,30/720,3.7/700)
    
    # Perform fit 
    if not detected: 
    # Slow line fit 
        fits,windows_warped,nonzeroy,nonzerox = lines_fits(warped,undisted,
                                                           Minv,
                                                           left_line, right_line, 
                                                           histogram_cfg,windows_cfg,curverad_cfg)
        
        fits_warped, out_img_fit = viz_fits(warped,nonzerox,nonzeroy,fits,windows_cfg)
        
        curverads, detected = lines_cr(warped, fits, left_line, right_line, curverad_cfg, detected)
    # add moving average of line fit coefficients    
        left_fit = fits['left_fit']
        right_fit = fits['right_fit']
        left_cur = curverads['left_curverad']
        right_cur = curverads['right_curverad']
        left_line.add_ABC_Cur(left_fit,left_cur) 
        right_line.add_ABC_Cur(right_fit,right_cur) 
        detected = True  # slow line fit always detects the line 

    else:
    # Fast line fit 
        left_fit,left_Cur = left_line.get_LastN_Avg_ABC_Cur() 
        right_fit,right_Cur = right_line.get_LastN_Avg_ABC_Cur() 
        fits, nonzerox, nonzeroy, detected = quick_fit(warped, left_fit, right_fit, left_line, right_line, windows_cfg, detected) 
        
        fits_warped, out_img_fit = viz_fits(warped,nonzerox,nonzeroy,fits,windows_cfg)
        
        # Only make updates if we detected lines in current frame 
        if fits is not None:
            
            #each 10 clip get a change back to slow fit from last bad fit 
            #slow_fit_again += 1
            #if slow_fit_again >10:
                #detected = False
                #slow_fit_again = 0
            
            left_fit = fits['left_fit']
            right_fit = fits['right_fit']
            
            curverads, detected = lines_cr(warped, fits, left_line, right_line, curverad_cfg, detected)
            left_cur = curverads['left_curverad']
            right_cur = curverads['right_curverad']
            
            left_fit = left_line.add_ABC_Cur(left_fit,left_cur) 
            right_fit = right_line.add_ABC_Cur(right_fit,right_cur)
            
        else: 
            detected = False 

    offset_vehicle = offset_v(warped,fits,curverad_cfg)

    # Perform final visualization on top of original undistorted image 
    result = final_output(warped,undisted,fits,Minv,curverads,offset_vehicle)
    
    result_w = result.shape[1]
    result_h = result.shape[0]
    
    rw = warped.shape[1]//5
    rh = warped.shape[0]//5
    
    combined_binary = cv2.resize(combined_binary, (rw, rh))
    combined_binary = np.dstack((combined_binary, combined_binary, combined_binary))*255
    
    roi_combined_binary = cv2.resize(roi_combined_binary, (rw, rh))
    roi_combined_binary = np.dstack((roi_combined_binary, roi_combined_binary, roi_combined_binary))*255
    
    warped = cv2.resize(warped, (rw, rh))
    warped = np.dstack((warped, warped, warped))*255
    
    fits_warped = cv2.resize(fits_warped, (rw, rh))

    
    result[10 : rh+10,
           result_w*4//5 : result_w*4//5+rw,
           0 : 3] = combined_binary
    
    result[2*10+rh : 10*2+2*rh, 
           result_w*4//5 : result_w*4//5+rw, 
           0 : 3] = roi_combined_binary

    result[3*10+2*rh : 3*10+3*rh, 
           result_w*4//5 : result_w*4//5+rw, 
           0 : 3] = warped

    result[4*10+3*rh : 4*10+4*rh,
           result_w*4//5 : result_w*4//5+rw,
           0 : 3] = fits_warped

    return result 

def line_fit_video_proc(input_file, output_file): 

    video = VideoFileClip(input_file).subclip(0, 20)
    line_fit_video_output = video.fl_image(line_fit_video) 
    line_fit_video_output.write_videofile(output_file, audio=False) 
