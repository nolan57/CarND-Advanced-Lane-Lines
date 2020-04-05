import numpy as np
import cv2
import pickle

def obj_img_pts(objp,fnames,ptsn,tosave = './'):
    
    # Arrays to store object points and image points from all the images.
    #3d points in real world space
    objpoints = []
    #2d points in image plane.
    imgpoints = []
    
    #for saving the objpoints and imgpoints
    points_pickle = {}
    log = []
    
    imgs={}  
    for fname in fnames:
        img = cv2.imread(fname)

        if '/' in fname:
            fname = fname.split('/')[-1]
        
        imgs[fname]=img
        
    isize=(img.shape[1],img.shape[0])
        
    # Step through the list and search for chessboard corners    
    for fname in imgs.keys():       
        img=imgs[fname]
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, ptsn, None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

        else:
            print('NO CORNERS FOUND ON: '+fname)
            
    points_pickle['objpoints'] = objpoints
    points_pickle['imgpoints'] = imgpoints
    pickle.dump(points_pickle,open('./obj_img_points.pickle','wb'))

    print('calibrating Camera is DONE!')
        
    return objpoints,imgpoints,isize

def Matrix_Coeffs(objpoints,imgpoints,isize):
    
    retval,comeraMatrix,distCoeffs,rvecs,tvexs = cv2.calibrateCamera(objpoints,imgpoints,isize,None,None)
    
    calibrateCamera_pickle={}
    
    calibrateCamera_pickle['retval']=retval
    calibrateCamera_pickle['comeraMatrix']=comeraMatrix
    calibrateCamera_pickle['distCoeffs']=distCoeffs
    calibrateCamera_pickle['rvecs']=rvecs
    calibrateCamera_pickle['tvexs']=tvexs
    
    pickle.dump(calibrateCamera_pickle,open('./calibrateCamera.pickle','wb'))
    
    return comeraMatrix,distCoeffs

# Define a function that takes an chessboard, number of x and y points, 
# camera matrix and distortion coefficients
def corners_unwarp(img, nx, ny, mtx, dist,offset=100):
    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Convert undistorted image to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    # Search for corners in the grayscaled image
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
    warped=None
    M=None
    if ret == True:
        # If we found corners, draw them! (just for fun)
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        # Choose offset from image corners to plot detected corners
        # This should be chosen to present the result at the proper aspect ratio
        # My choice of 100 pixels is not exact, but close enough for our purpose here
        
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])

        # For source points I'm grabbing the outer four detected corners
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result 
        # again, not exact, but close enough for our purposes
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                     [img_size[0]-offset, img_size[1]-offset], 
                                     [offset, img_size[1]-offset]])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, M, img_size)

    # Return the resulting image and matrix
    return undist,warped, M