import pickle
import cv2
import numpy as np

def abs_sobel_thresh(img,orient='x',sobel_kernel=3,thresh=(0,255)):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if orient == 'x': 
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel))
    if orient == 'y': 
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel))

    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel)) 
    abs_binary = np.zeros_like(scaled_sobel) 

    abs_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1 
    
    return abs_binary

def dir_sobel_thresh(img,sobel_kernel=3,thresh=(0,np.pi/2)):

    img=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
    sobelx=cv2.Sobel(img, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
    sobely=cv2.Sobel(img, cv2.CV_64F, 0, 1,ksize=sobel_kernel)    
    dir_sobel=np.arctan2(np.absolute(sobely),np.absolute(sobelx))  
    
    dir_binary=np.zeros_like(dir_sobel)
    dir_binary[(dir_sobel>=thresh[0])&(dir_sobel<=thresh[1])]=1
    
    return dir_binary

def mag_sobel_thresh(img,sobel_kernel=3,thresh=(0,255)):

    img=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    sobelx=cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely=cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)    
    mag_sb=np.sqrt(sobelx**2+sobely**2)
    
    mag_sb=np.uint8(255*mag_sb/np.max(mag_sb))    
    mag_binary=np.zeros_like(mag_sb)
    mag_binary[(mag_sb>=thresh[0])&(mag_sb<=thresh[1])]=1
    
    return mag_binary
#---------------------------------------------------------------------------------------------------------------
def hls_thresh(img,thresh=None):
    
    hls=cv2.cvtColor(img,cv2.COLOR_RGB2HLS).astype(np.float)
    S_channel=hls[:,:,2]
    S_binary=np.zeros_like(S_channel)
    #S_binary[(S_channel>=thresh[0])&(S_channel<=thresh[1])]=1
    S_binary[(S_channel>=80)&(S_channel<=255)]=1
    
    hls=cv2.cvtColor(img,cv2.COLOR_RGB2HLS).astype(np.float)
    L_channel=hls[:,:,1]
    L_binary=np.zeros_like(L_channel)
    #L_binary[(S_channel>=thresh[0])&(S_channel<=thresh[1])]=1
    L_binary[(S_channel>=175)&(S_channel<=255)]=1
    
    HSL_binary = np.zeros_like(S_binary)
    HSL_binary[(S_binary == 1) | (L_binary == 1)] = 1
    
    return HSL_binary
# ----------------------------------------------------------------------------------------------------------------
def yellow_white(img):

    HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # for yellow
    yellow = cv2.inRange(HSV, (20, 100, 100), (50, 255, 255))

    # for white
    sensitivity_1 = 68
    white_1 = cv2.inRange(HSV, (0, 0, 255 - sensitivity_1), (255, 20, 255))

    sensitivity_2 = 60
    HSL = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    white_2 = cv2.inRange(HSL, (0, 255 - sensitivity_2, 0), (255, 255, sensitivity_2))

    white_3 = cv2.inRange(img, (200, 200, 200), (255, 255, 255))

    #yellow_white_layer = np.zeros_like(img)

    yellow_white_layer = yellow | white_1 | white_2 | white_3

    #return yellow_white_layer, yellow, white_1, white_2, white_3 #for debug
    return yellow_white_layer
#----------------------------------------------------------------------------------------------------------------
def color_gradient(img):

    #HSL = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)

    #S_channel = HSL[:, :, 2]
    #L_channel = HSL[:, :, 1]

    #abs_binary_S = abs_sobel_thresh(S_channel, orient='x', thresh=(30, 100))
    #abs_binary_L = abs_sobel_thresh(L_channel, orient='x', thresh=(20, 100))
    #abs_binary = abs_sobel_thresh(img, orient='x', thresh=(5, 100))
    
    #dir_binary = dir_sobel_thresh(img, sobel_kernel=15, thresh=(np.pi/6, np.pi/3))
    #mag_binary = mag_sobel_thresh(img, sobel_kernel=3, thresh=(50, 255))
    HSL_binary = hls_thresh(img)
    
    yellow_white_layer = yellow_white(img)
    
    #abs_ywl = np.logical_or(abs_binary, yellow_white_layer)
    
    #color_binary = np.dstack(( np.zeros_like(abs_binary), abs_binary, HSL_binary))
    #color_binary = np.dstack((np.zeros_like(abs_binary), abs_binary, abs_binary))

    #abs_ywl = np.logical_or(abs_binary, yellow_white_layer)
    combined_binary = np.zeros_like(HSL_binary)
    combined_binary[(yellow_white_layer == True)] = 1
    #combined_binary[(combined_binary == 1) | ((dir_binary ==1 ) & (mag_binary ==1 ))] = 1
    
    #combined_binary[(abs_binary_S == 1) & (abs_binary_L == 1)] = 1
    #combined_binary[(abs_binary_L == 1)] = 1
    
    #mag_a_dir_binary = np.copy(combined_binary)

    combined_binary[(combined_binary == 1) | (HSL_binary == 1)] = 1

    #combined_binary[abs_binary == 1] = 1

    #combined_binary = combined_binary | abs_ywl
    
    #return combined_binary, abs_binary, dir_binary, mag_binary, mag_a_dir_binary, HSL_binary, color_binary
    return combined_binary, yellow_white_layer, HSL_binary #abs_binary #abs_binary_L, color_binary
#-----------------------------------------------------------------------------------------------------------------
#i set a roi of warped images as p1 befor for line fit
def roi_mask(img):
    
    h=img.shape[0]
    w=img.shape[1]
    mid=w//2
    
    #below roi just let lane line on screen
    roi_vtx = np.array([[(0, 720), (mid-100, 400), 
                     (mid+100, 400), (w, 720)]])
    
    mask = np.zeros_like(img)
    mask_color = 255
    cv2.fillPoly(mask, roi_vtx, mask_color)
    masked_img = cv2.bitwise_and(img, mask)
    
    #and below roi try to clear away nosie between lane line
    roi_vtx = np.array([[(400, 760), (640, 600), 
                     (640, 600), (900, 760)]])
        
    mask = np.ones_like(img)
    mask_color = 0
    cv2.fillPoly(mask, roi_vtx, mask_color)
    masked_img = cv2.bitwise_and(masked_img, mask)
    
    return masked_img
#-------------------------------------------------------------------------------------------------------------------
#the pipeline function to preprocess input for lane lines fit later
def preproc_images(img,comeraMatrix,distCoeffs,M):

    undisted = cv2.undistort(img,comeraMatrix,distCoeffs,None,comeraMatrix)
    
        
    chimg = np.copy(undisted)
    ch1 = chimg[:, :, 0]
    ch2 = chimg[:, :, 1]
    ch3 = chimg[:, :, 2]
    chh1 = cv2.equalizeHist(ch1)
    chh2 = cv2.equalizeHist(ch2)
    chh3 = cv2.equalizeHist(ch3)
    hist_img = np.dstack((chh1, chh2, chh3))
    
    #combined_binary, abs_binary, dir_binary, mag_binary, mag_a_dir_binary, HSL_binary, color_binary = color_gradient(undisted)
    combined_binary, yellow_white_layer, HSL_binary = color_gradient(hist_img*255)

    dsize=(combined_binary.shape[1],combined_binary.shape[0])
    
    roi_combined_binary=roi_mask(combined_binary)
    
    warped = cv2.warpPerspective(roi_combined_binary,M,dsize,
                              flags = cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=0)
    
    return undisted, warped, combined_binary*255, roi_combined_binary, HSL_binary