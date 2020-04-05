import cv2
import numpy as np
from preproc_images import preproc_images

def lines_fits(warped,undisted,Minv, left_line, right_line, histogram_cfg, windows_cfg,curverad_cfg):
    
    top,bottom,left,right = histogram_cfg
    nwindows,margin,minpix = windows_cfg
    y_eval,ym_per_pix,xm_per_pix = curverad_cfg
    
    histogram = np.sum(warped[top:bottom,left:right], axis=0)
    # Create an output image to draw on and  visualize the result
    windows_warped = np.dstack((warped, warped, warped))*255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint-100])
    rightx_base = np.argmax(histogram[midpoint+100:]) + midpoint+100

    # Set height of windows
    window_height = np.int(warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped.shape[0] - (window+1)*window_height
        win_y_high = warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(windows_warped,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(windows_warped,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) 
                          & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) 
                           & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
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
    
        #if current none pixels fited,use last avg coeffients
    if (leftx.size == 0) or (lefty.size == 0) or (rightx.size == 0) or (righty.size == 0):
        left_fit, left_avg_Cur = left_line.get_LastN_Avg_ABC_Cur()
        right_fit, right_avg_Cur = right_line.get_LastN_Avg_ABC_Cur()
        detected = False
    else:
        # Fit a second order polynomial to each
        #left_fit = np.polyfit(lefty, leftx, 3)
        #right_fit = np.polyfit(righty, rightx, 3)
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

    # Fit a second order polynomial to each
    #left_fit = np.polyfit(lefty, leftx, 2)
    #right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    #left_fitx = left_fit[0]*(ploty**3) + left_fit[1]*(ploty**2) + left_fit[2]*ploty + left_fit[3]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    #right_fitx = right_fit[0]*(ploty**3) + right_fit[1]*(ploty**2) + right_fit[2]*ploty + right_fit[3]
    
    fits={}
    fits['left_lane_inds'] = left_lane_inds
    fits['right_lane_inds'] = right_lane_inds
    fits['leftx'] = leftx #left line pixel(x) positions detected
    fits['rightx'] = rightx #right line pixel(x) positions detected
    fits['lefty'] = lefty #left line pixel(y) positions detected
    fits['righty'] = righty #right line pixel(y) positions detected
    fits['left_fit'] = left_fit #left line coefficients fited by positions detected
    fits['right_fit'] = right_fit #right line coefficients fited by positions detected
    fits['ploty'] = ploty #linear space to fit line
    fits['left_fitx'] = left_fitx #left x value gernerated by fit
    fits['right_fitx'] = right_fitx #right x value gernerated by fit
    
    return fits,windows_warped,nonzeroy,nonzerox
    
#----------------------------------------------------------------------------------------------------------------------------------
def quick_fit(warped, left_fit, right_fit,left_line,right_line,windows_cfg, detected):
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    nwindows,margin,minpix = windows_cfg
    
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    #left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**3) + left_fit[1]*(nonzeroy**2) + left_fit[2]*nonzeroy + left_fit[3] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**3) + left_fit[1]*(nonzeroy**2) + left_fit[2]*nonzeroy + left_fit[3] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))
    #right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**3) + right_fit[1]*(nonzeroy**2) + right_fit[2]*nonzeroy + right_fit[3] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**3) + right_fit[1]*(nonzeroy**2) + right_fit[2]*nonzeroy + right_fit[3] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    #if current none pixels fited,use last avg coeffients
    if (leftx.size == 0) or (lefty.size == 0) or (rightx.size == 0) or (righty.size == 0):
        left_fit, left_avg_Cur = left_line.get_LastN_Avg_ABC_Cur()
        right_fit, right_avg_Cur = right_line.get_LastN_Avg_ABC_Cur()
        detected = True
    else:
        # Fit a second order polynomial to each
        #left_fit = np.polyfit(lefty, leftx, 3)
        #right_fit = np.polyfit(righty, rightx, 3)
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    #left_fitx = left_fit[0]*(ploty**3) + left_fit[1]*(ploty**2) + left_fit[2]*ploty + left_fit[3]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    #right_fitx = right_fit[0]*(ploty**3) + right_fit[1]*(ploty**2) + right_fit[2]*ploty + right_fit[3]
    
    fits={}
    fits['left_lane_inds'] = left_lane_inds
    fits['right_lane_inds'] = right_lane_inds
    fits['leftx'] = leftx #left line pixel(x) positions detected
    fits['rightx'] = rightx #right line pixel(x) positions detected
    fits['lefty'] = lefty #left line pixel(y) positions detected
    fits['righty'] = righty #right line pixel(y) positions detected
    fits['left_fit'] = left_fit #left line coefficients fited by positions detected
    fits['right_fit'] = right_fit #right line coefficients fited by positions detected
    fits['ploty'] = ploty #linear space to fit line
    fits['left_fitx'] = left_fitx #left x value gernerated by fit
    fits['right_fitx'] = right_fitx #right x value gernerated by fit
    
    return fits, nonzerox, nonzeroy, detected

#----------------------------------------------------------------------------------------------------------------------------------
#lefty,leftx,righty,rightx,ym_per_pix,xm_per_pix,y_eval,warped
    # Fit new polynomials to x,y in world space
def lines_cr(warped,fits, left_line, right_line, curverad_cfg, detected):
    
    lefty = fits['lefty']
    leftx = fits['leftx']
    righty = fits['rightx']
    rightx = fits['righty']

    #if current none pixels fited,use last avg coeffients
    if (leftx.size == 0) or (lefty.size == 0) or (rightx.size == 0) or (righty.size == 0):
        left_fit, left_avg_Cur = left_line.get_LastN_Avg_ABC_Cur()
        right_fit, right_avg_Cur = right_line.get_LastN_Avg_ABC_Cur()
        
        curverads = {}
        curverads['left_curverad'] = left_avg_Cur
        curverads['right_curverad'] = right_avg_Cur
        detected = True
        
        return curverads, detected
        
    y_eval,ym_per_pix,xm_per_pix = curverad_cfg
    
    
    #left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 3)
    #right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 3)
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    #left_curverad = ((1 + (3*left_fit_cr[0]*(y_eval*ym_per_pix)**2 + 2*left_fit_cr[1]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(6*left_fit_cr[0]*y_eval*ym_per_pix + 2*left_fit_cr[1]*y_eval*ym_per_pix)
    
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    #right_curverad = ((1 + (3*right_fit_cr[0]*(y_eval*ym_per_pix)**2 + 2*right_fit_cr[1]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(6*right_fit_cr[0]*y_eval*ym_per_pix + 2*right_fit_cr[1]*y_eval*ym_per_pix)
    
    curverads = {}
    curverads['left_curverad'] = left_curverad
    curverads['right_curverad'] = right_curverad
                  
    return curverads, detected
#------------------------------------------------------------------------------------------------------------------------------------
#left_fit,right_fit,y_eval,xm_per_pix,midpoint,warped
def offset_v(warped,fits,curverad_cfg):
                  
    left_fit = fits['left_fit']
    right_fit = fits['right_fit']
    y_eval,ym_per_pix,xm_per_pix = curverad_cfg
    midpoint = warped.shape[1]//2
                      
    #left_line_pos = left_fit[0]*y_eval**3 + left_fit[1]*y_eval**2 + left_fit[2]*y_eval + left_fit[3]
    #right_line_pos = right_fit[0]*y_eval**3 + right_fit[1]*y_eval**2 + right_fit[2]*y_eval + right_fit[3]
    left_line_pos = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_line_pos = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    
    center_between_lines = (right_line_pos+left_line_pos)//2
    offset_vehicle = (midpoint-center_between_lines)*xm_per_pix
                  
    return offset_vehicle
#-------------------------------------------------------------------------------------------------------------------------------------
#warped,nonzeroy,nonzerox,left_fitx,margin,ploty,right_fitx,
def viz_fits(warped,nonzerox,nonzeroy,fits,windows_cfg):
                  
    left_fitx = fits['left_fitx']
    right_fitx = fits['right_fitx']
    ploty = fits['ploty']
    left_lane_inds = fits['left_lane_inds']
    right_lane_inds = fits['right_lane_inds']
    nwindows,margin,minpix = windows_cfg
                  
    # Create an image to draw on and an image to show the selection window
    out_img_fit = np.dstack((warped, warped, warped))*255
    window_img = np.zeros_like(out_img_fit)
    # Color in left and right line pixels
    out_img_fit[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img_fit[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    fits_warped = cv2.addWeighted(out_img_fit, 1, window_img, 0.3, 0)
    
    return fits_warped, out_img_fit
#--------------------------------------------------------------------------------------------------------------------------------------
def final_output(warped,undisted,fits,Minv,curverads,offset_vehicle):
                  
    left_fitx  =fits['left_fitx']
    right_fitx = fits['right_fitx']
    ploty = fits['ploty']
                  
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    left_radius = curverads['left_curverad']/1000.
    right_raius = curverads['right_curverad']/1000.
    #avg_curverad = (curverads['left_curverad']+curverads['right_curverad'])/2/1000
    avg_curverad = (left_radius+right_raius)/2.
    
    if avg_curverad <1.0:
    # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    else:
         cv2.fillPoly(color_warp, np.int_([pts]), (255,0, 0))        
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (warped.shape[1], warped.shape[0]))
    # Combine the result with the original image 
    result_fit = cv2.addWeighted( undisted, 1, newwarp, 0.3, 0) 
    
    label_left_curverad = 'Radius of left line: %.2f km' % left_radius
    label_right_curverad = 'Radius of right line: %.2f km' % right_raius
    label_avg_curverad = 'Avg Radius of curvature: %.2f km' % avg_curverad
    
    output = cv2.putText(result_fit, label_left_curverad, (30,40), 0, 0.7, (255,255,255), 2, cv2.LINE_AA) 
    output = cv2.putText(output, label_right_curverad, (640,40), 0, 0.7, (255,255,255), 2, cv2.LINE_AA)
    
    if (left_radius>2.) and (right_raius>2.):
        output = cv2.putText(output, 'STRAIGHT LANE!', (400,70), 0, 1, (0,255,0), 2, cv2.LINE_AA)
    else:
        output = cv2.putText(output, label_avg_curverad, (300,70), 0, 1, (255,255,255), 2, cv2.LINE_AA)
    
    label_offset_vehicle = 'Vehicle offset from lane center: %.2f m' % offset_vehicle
    output = cv2.putText(output, label_offset_vehicle, (30,100), 0, 1, (255,255,255), 2, cv2.LINE_AA)
    
    return output
#------------------------------------------------------------------------------------------------------------------------------------------
#define a line class to smooth the fit line
class Line():
    
    def __init__(self,N_LastToAvg):
        #how many last n line to average
        self.N_LastToAvg = N_LastToAvg
        #assume first not detected
        self.detected = False
        #fit coefficients of x = A*y**2+B*y+C
        self.A = []
        self.B = []
        self.C = []
        #self.D = []
        #average of last n fit coefficients
        self.avg_A = 0.
        self.avg_B = 0.
        self.avg_C = 0.
        #self.avg_D = 0.
        #last n line curverads
        self.Cur = []
        #average of last n line curverads
        self.avg_Cur = 0.

        
    def get_LastN_Avg_ABC_Cur(self):
        
        #return (self.avg_A,self.avg_B,self.avg_C, self.avg_D), self.avg_Cur
        return (self.avg_A,self.avg_B,self.avg_C), self.avg_Cur
    
    def add_ABC_Cur(self,ABC,Cur):
        
        full = len(self.A) >= self.N_LastToAvg
        
        self.A.append(ABC[0])
        self.B.append(ABC[1])
        self.C.append(ABC[2])
        #self.D.append(ABC[3])
        self.Cur.append(Cur)
        
        if full:
            _ = self.A.pop(0)
            _ = self.B.pop(0)
            _ = self.C.pop(0)
            #_ = self.D.pop(0)
            _ = self.Cur.pop(0)
        
        self.avg_A = np.mean(self.A)
        self.avg_B = np.mean(self.B)
        self.avg_C = np.mean(self.C)
        #self.avg_D = np.mean(self.D)
        self.avg_Cur = np.mean(self.Cur)

   