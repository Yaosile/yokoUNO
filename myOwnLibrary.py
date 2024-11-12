import numpy as np
from numpy import asanyarray as ana
# import numba

scaling = 1
cameraWidth = 3264//scaling
cameraHeight = 2464//scaling

dist = ana([-0.0639733628476694, -0.059022840140777, 0, 0, 0.0238818089164303])
mtx = ana([
    [1.734239392051136E3,0,1.667798059392088E3],
    [0,1.729637617052701E3,1.195682065165660E3],
    [0,0,1],
])

def linear_interpolate(v1, v2, fraction):
    return (1 - fraction) * v1 + fraction * v2

def bilinear_interpolation(image, x, y):
    x1 = min(int(x), image.shape[1]-1)  # floor of x
    y1 = min(int(y), image.shape[0]-1)  # floor of y
    x2 = min(x1 + 1, image.shape[1] - 1)  # ensure it stays within bounds
    y2 = min(y1 + 1, image.shape[0] - 1)

    # Calculate the fractional part of x and y
    x_frac = x - x1
    y_frac = y - y1

    # Get pixel values at the corners
    Q11 = image[y1, x1]
    Q12 = image[y2, x1]
    Q21 = image[y1, x2]
    Q22 = image[y2, x2]

    # Perform linear interpolation in the x direction
    R1 = linear_interpolate(Q11, Q21, x_frac)
    R2 = linear_interpolate(Q12, Q22, x_frac)

    # Perform linear interpolation in the y direction
    P = linear_interpolate(R1, R2, y_frac)

    return P

def scaleImage(image, width, height):
    if image.shape[2] > 1:
        temp = np.zeros((height, width, image.shape[2]))
    else:
        temp = np.zeros((height,width))
    xScale = image.shape[1]/width
    yScale = image.shape[0]/height
    for i in range(width):
        for j in range(height):
            x = i*xScale
            y = j*yScale
            temp[j,i] = bilinear_interpolation(image, x, y)

    return temp

def UVGrid(frame: np.ndarray):
    x,y = np.linspace(0,frame.shape[0]-1,frame.shape[0]), np.linspace(0,frame.shape[1]-1,frame.shape[1])
    u,v = np.meshgrid(x,y)
    return u,v

def rot90(x: np.ndarray, n=1):
    n = n%4
    if n == 0:
        return x
    if n>1:
        x = rot90(x, n-1)
    dim = len(x), len(x[0])
    rotated = np.zeros((dim[1], dim[0]))
    for i in range(dim[0]):
        for j in range(dim[1]):
            rotated[dim[1]-j-1, i] = x[i, j]
    return rotated

def arucoCorners(frame: np.ndarray):
    '''MY OWN'''
    shape = frame.shape
    u,v = UVGrid(frame)
    max = int(np.max(u+v))
    original = 255-frame.copy()
    # final = np.zeros_like(original)
    positions = []
    for j in range(4):
        output = rot90(original, j)
        multiply = np.ones_like(frame)
        prev = np.average(output)
        for i in range(max//2):
            multiply[u+v < i] = 0
            output = output * multiply
            if np.average(output) != prev:
                multiply[u+v > i] = 0
                output = output * multiply
                positions.append((np.argmax(output)%frame.shape[1], np.argmax(output)//frame.shape[1]))
                break
    positions[1] = shape[0]-positions[1][1], positions[1][0]
    positions[2] = shape[0]-positions[2][0], shape[1]-positions[2][1]
    positions[3] = positions[3][1], shape[1]-positions[3][0]
    return positions


def pixelToCartesian(px,py,imageWidth,imageHeight): 
    '''MY OWN'''
    boardDimensions = (605, 517) #height, width in mm
    y = (imageHeight - py)*(boardDimensions[0]/imageHeight)
    x = (px - imageWidth/2)*(boardDimensions[1]/imageWidth)
    return x,y

def cartesianToScara(x,y):
    '''MY OWN'''
    #Right side dimensions
    xOffset = 150
    L1 = 250+14+14
    L2 = xOffset + L1

    #Right Motor
    d = np.sqrt((x-xOffset)**2 + y**2) #calculating the distance from the motor to point
    a = (L2**2-L1**2+d**2)/(2*d)
    h = np.sqrt(L2**2-a**2)
    if False:
        xr = x + a*(xOffset-x)/d + h*(-y)/d
        yr = y + a*(-y)/d + h*(x-xOffset)/d
    else:
        xr = x + a*(xOffset-x)/d - h*(-y)/d
        yr = y + a*(-y)/d - h*(x-xOffset)/d
    theta1r = np.arctan2(yr, xr-xOffset)
    theta2r = np.arctan2(y-yr, x-xr)

    #Left side dimensions
    xOffset = 150
    L1 = 250+14+14
    L2 = xOffset + L1
    #Left Motor
    xOffset = -xOffset
    d = np.sqrt((x-xOffset)**2 + y**2)
    a = (L2**2-L1**2+d**2)/(2*d)
    h = np.sqrt(L2**2-a**2)
    if True:
        xl = x + a*(xOffset-x)/d + h*(-y)/d
        yl = y + a*(-y)/d + h*(x-xOffset)/d
    else:
        xl = x + a*(xOffset-x)/d - h*(-y)/d
        yl = y + a*(-y)/d - h*(x-xOffset)/d
    theta1l = np.arctan2(yl, xl-xOffset)
    theta2l = np.arctan2(y-yl, x-xl)

    return theta1l, theta1r

def normalise(array, minOut = 0, maxOut = 255):
    '''MY OWN but insignificant'''
    array = array.copy()
    minIn = np.min(array)
    maxIn = np.max(array)
    array -= minIn
    array /= maxIn
    array *= maxOut
    array += minOut
    return array

def rgb2gray(image):
    '''Standard'''
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

def rgb2hsv(image, Calculations = 'HSV'):
    '''Standard'''
    output = np.zeros_like(image).astype(float)
    Cmax = np.max(image, axis = 2) #not normalised
    Cmin = np.min(image, axis = 2) #not normalised
    maxIndex = np.argmax(image, axis=2)
    delta = Cmax - Cmin #not normalised
    
    #Hue calculations
    if 'H' in Calculations:
        output[:,:,0][(delta != 0) & (maxIndex == 0)] = (1/6)*(((image[:,:,1][(delta != 0) & (maxIndex == 0)] - image[:,:,2][(delta != 0) & (maxIndex == 0)])/(delta[(delta != 0) & (maxIndex == 0)]))%6)
        output[:,:,0][(delta != 0) & (maxIndex == 1)] = (1/6)*(((image[:,:,2][(delta != 0) & (maxIndex == 1)] - image[:,:,0][(delta != 0) & (maxIndex == 1)])/(delta[(delta != 0) & (maxIndex == 1)]))+2)
        output[:,:,0][(delta != 0) & (maxIndex == 2)] = (1/6)*(((image[:,:,0][(delta != 0) & (maxIndex == 2)] - image[:,:,1][(delta != 0) & (maxIndex == 2)])/(delta[(delta != 0) & (maxIndex == 2)]))+4)

    #Satuation calculations
    if 'S' in Calculations:
        output[:,:,1][Cmax != 0] = delta[Cmax != 0]/Cmax[Cmax != 0]

    #Value calculations
    if 'V' in Calculations:
        output[:,:,2] = Cmax/255
    return output

def threshHold(arrayLike, hold):
    '''Standard'''
    temp = np.zeros(arrayLike.shape)
    threshHeld = np.where(arrayLike > hold)
    temp[threshHeld] = 255
    return temp

def onCardLines(frame: np.ndarray):
    '''MY OWN'''
    x_middle = np.mean(frame, axis=0)
    y_middle = np.mean(frame, axis=1)
    x_middle = np.argmax(x_middle)
    y_middle = np.argmax(y_middle)
    return x_middle, y_middle

# def scanLines(frame: np.ndarray,x,y):
#     top = np.mean(frame[:y,:],axis=1)
#     bot = np.mean(frame[y:,:],axis=1)
#     left = np.mean(frame[:,:x],axis=0)
#     right = np.mean(frame[:,x:],axis=0)

def boundingBox(frame):
    '''MY OWN'''
    frame = frame.copy()
    valid = np.where(frame > 254)
    t = valid[0][0]
    b = valid[0][-1]
    l = np.min(valid[1])
    r = np.max(valid[1])

    return t,b,l,r

def getRadius(frame, centreX, centreY):
    '''MY OWN'''
    x,y = np.meshgrid(np.arange(frame.shape[1])-centreX, np.arange(frame.shape[0])-centreY)
    r,b = (x**2 + y**2)**0.5, np.atan2(y,x)
    values = []
    i = 50
    while True:
        i+= 1
        condition = (r < i+50) & (r>i)
        if np.sum(frame[condition]) == 0:
            break
    return i

def rotate(frame, theta):
    '''MY OWN'''
    "Rotates an image, but only a square one"
    radius = frame.shape[0]/2
    xo,yo = np.meshgrid(np.arange(radius*2)-radius, np.arange(radius*2)-radius)
    r,b = (xo**2 + yo**2)**0.5, np.atan2(yo,xo)
    xd,yd = ((r*np.cos(b+theta))+radius), ((r*np.sin(b+theta))+radius)
    xd,yd = xd.astype(int), yd.astype(int)
    xd[xd>=radius*2] = radius*2-1
    xd[xd<0] = 0
    yd[yd>=radius*2] = radius*2-1
    yd[yd<0] = 0
    rotated = np.zeros_like(frame)
    rotated = frame[yd,xd]
    return rotated

def isolateCard(frame, originalImage):
    '''MY OWN'''
    # frame = frame.copy()
    # originalImage = originalImage.copy()
    t,b,l,right = boundingBox(frame)
    cx,cy = midPoint(t,b,l,right)
    r = getRadius(frame, cx,cy)
    theta = getRotation(frame, cx,cy,r)
    temp = rotate(frame[-r+cy:r+cy, -r+cx:r+cx], theta)
    t,b,l,right = boundingBox(temp)
    temp = rotate(originalImage[-r+cy:r+cy, -r+cx:r+cx], theta)
    # temp[t,:] = [255,0,0]
    # temp[b,:] = [255,0,0]
    # temp[:,l] = [255,0,0]
    # temp[:,r] = [255,0,0]
    # temp[t,:] = 255
    # temp[b,:] = 255
    # temp[:,l] = 255
    # temp[:,r] = 255
    # return temp[t-4:b-4,l-4:right-4]
    return temp[t:b,l:right]

def getRotation(frame, centreX, centreY, radius):
    '''MY OWN'''
    radius += 10
    temp = frame[-radius+centreY:radius+centreY, -radius+centreX:radius+centreX]
    xo,yo = np.meshgrid(np.arange(radius*2)-radius, np.arange(radius*2)-radius)
    r,b = (xo**2 + yo**2)**0.5, np.atan(yo,xo)
    change = np.pi/6
    sumChange = change
    rotated = temp.copy()
    for i in range(30):
        top,bot,left,right = boundingBox(rotated)
        diff = (right-left)/(bot-top)
        xd,yd = ((r*np.cos(b+sumChange))+radius), ((r*np.sin(b+sumChange))+radius)
        xd,yd = xd.astype(int), yd.astype(int)
        xd[xd>=radius*2] = radius*2-1
        xd[xd<0] = 0
        yd[yd>=radius*2] = radius*2-1
        yd[yd<0] = 0
        rotated = temp[yd,xd]
        # print(i,sumChange*180/np.pi,sep='\t')
        top,bot,left,right = boundingBox(rotated)
        newDiff = (right-left)/(bot-top)
        if newDiff > diff:
            change = -change
        change *= 0.75
        sumChange += change

    return sumChange



def scanLines(frame: np.ndarray,x,y,thresholdAmount):
    '''MY OWN'''
    top = np.mean(frame[:y,:],axis=1)
    bot = np.mean(frame[y:,:],axis=1)
    left = np.mean(frame[:,:x],axis=0)
    right = np.mean(frame[:,x:],axis=0)
    top = largestConsecutiveSet(np.where(top < thresholdAmount))
    bot = largestConsecutiveSet(np.where(bot < thresholdAmount))
    left = largestConsecutiveSet(np.where(left < thresholdAmount))
    right = largestConsecutiveSet(np.where(right < thresholdAmount))
    top = top[-1] if len(top) > 1 else 0
    bot = bot[0] if len(bot) > 1 else 0
    left = left[-1] if len(left) > 1 else 0
    right = right[0] if len(right) > 1 else 0

    top = top if top < frame.shape[1] else frame.shape[1] - 1
    bot = bot + y if bot + y < frame.shape[1] else frame.shape[1] - 1
    right = right + x if right + x < frame.shape[0] else frame.shape[0] - 1
    left = left if left < frame.shape[0] else frame.shape[0] - 1
    return top,bot,left,right

def largestConsecutiveSet(array:np.ndarray):
    '''MY OWN'''
    array = array[0].copy()
    maxLength = 0
    currentLength = 1
    startIndex = 0
    maxStartIndex = 0
    for i in range(1,len(array)):
        if array[i] == array[i-1]+1:
            currentLength += 1
        else:
            if currentLength > maxLength:
                maxLength = currentLength
                maxStartIndex = startIndex
            currentLength = 1
            startIndex = i
            
    if currentLength > maxLength:
        maxLength = currentLength
        maxStartIndex = startIndex

    return array[maxStartIndex:maxStartIndex+maxLength]

def midPoint(top,bottom,left,right):
    '''Insignificant'''
    midX = (left+right)//2
    midY = (top+bottom)//2
    return midX, midY

def closestValueInSet(value, setOfValues):
    '''MY OWN but insignificant'''
    return np.argmin(np.abs(setOfValues - value))

def gaussianKernelGenerator(size, sigma = 2):
    '''Standard'''
    x,y = np.meshgrid(np.arange(size)-size/2+0.5, np.arange(size)-size/2+0.5)
    output = np.exp(-((x**2 + y**2)**0.5)/(2*(sigma**2)))
    output /= output.sum()
    return output

# @numba.jit(nopython = True)
def generateGrid(width, height):
    '''MY OWN'''
    xd = np.empty(shape = (height, width), dtype=int)
    yd = np.empty(shape = (height, width), dtype=int)
    for i in np.arange(width):
        for j in np.arange(height):
            xd[j, i] = i
            yd[j, i] = j
    return xd,yd

# @numba.jit(nopython = True)
def distortionMap(distotionCoefficients = dist, mtx = mtx, width = cameraWidth, height = cameraHeight):
    '''MY OWN and standard'''
    K1 = distotionCoefficients[0]
    K2 = distotionCoefficients[1]
    K3 = distotionCoefficients[4]

    xc = mtx[0][2]
    yc = mtx[1][2]
    fx = mtx[0][0]
    fy = mtx[1][1]
    xd,yd = np.meshgrid(np.arange(width), np.arange(height))

    xu = (xd - xc) / fx
    yu = (yd - yc) / fy
    r2 = xu**2 + yu**2
    xu = xu/(1 + K1*r2 + K2*(r2**2) + K3*(r2**3))
    yu = yu/(1 + K1*r2 + K2*(r2**2) + K3*(r2**3))
    xu = xu*fx + xc
    yu = yu*fy + yc
    yu -= yu.min()
    yu /= yu.max()
    yu *= height-1
    xu -= xu.min()
    xu /= xu.max()
    xu *= width-1

    coordinateMatrix = np.stack((yd,xd), axis=-1)
    xu = xu.astype(int)
    yu = yu.astype(int)
    mask = np.ones((height, width), dtype=bool)
    mask[yu, xu] = False
    inverse = np.zeros((height, width, 2), dtype=int)
    inverse[yu, xu] = coordinateMatrix[yd,xd]

    for i, row in enumerate(mask):
        for j, val in enumerate(row):
            if val and i < height-1 and j < width-1:
                    inverse[i,j] = inverse[i,j+1]

    yu = inverse[:,:,0]
    xu = inverse[:,:,1]

    return yu, xu

# def unwarpMap(src, dstWidth, dstHeight, imageWidth, imageHeight):
#     '''MY OWN and standard'''
#     if type(src) != np.ndarray:
#         src = ana(src)
#     dst = np.array([[0, 0], [0, dstHeight-1], [dstWidth-1, dstHeight-1], [dstWidth-1,0]])
#     H = find_homography(src, dst)
#     H_inv = np.linalg.inv(H)
#     boundBox = [imageWidth, imageHeight]

#     xd, yd = np.meshgrid(np.arange(boundBox[0]), np.arange(boundBox[1]))
#     coordinateMatrix = np.stack((yd, xd), axis=-1)
#     output = np.zeros((dstHeight, dstWidth, 2), dtype=int)
#     for i in range(output.shape[1]):
#         for j in range(output.shape[0]):
#             x, y, z = np.dot(H_inv, ana([i, j, 1]))
#             x, y = x/z, y/z
#             if 0 <= x and x < boundBox[0] and 0 <= y and y < boundBox[1]:
#                 output[j, i] = coordinateMatrix[int(y), int(x)]
#     return output[:,:,0], output[:,:,1]

def unwarpMap(src, dstWidth, dstHeight, imageWidth=cameraWidth, imageHeight=cameraHeight):
    '''MY OWN/REFINED BY GPT'''
    if type(src) != np.ndarray:
        src = ana(src)
        
    # Destination points (corners of the output image)
    dst = np.array([[0, 0], [0, dstHeight-1], [dstWidth-1, dstHeight-1], [dstWidth-1, 0]])
    
    # Compute homography matrix
    H = find_homography(src, dst)
    H_inv = np.linalg.inv(H)
    
    # Generate grid of destination image coordinates
    xd, yd = np.meshgrid(np.arange(dstWidth), np.arange(dstHeight))
    
    # Convert the grid into homogeneous coordinates (flatten the arrays)
    ones = np.ones_like(xd)
    dst_points = np.stack([xd.ravel(), yd.ravel(), ones.ravel()], axis=0)
    
    # Apply the inverse homography to the entire grid
    transformed_points = np.dot(H_inv, dst_points)
    
    # Normalize by the third (homogeneous) coordinate
    transformed_points /= transformed_points[2, :]
    
    # Get the transformed (x, y) coordinates
    x_transformed = transformed_points[0].reshape(dstHeight, dstWidth)
    y_transformed = transformed_points[1].reshape(dstHeight, dstWidth)
    
    # Clip the coordinates to be within the image bounds
    x_transformed = np.clip(x_transformed, 0, imageWidth - 1).astype(int)
    y_transformed = np.clip(y_transformed, 0, imageHeight - 1).astype(int)
    
    # Return the mapping of destination image coordinates to source image coordinates
    return y_transformed, x_transformed

# def unwarpMap3D(src, dstWidth, dstHeight, imageWidth, imageHeight, zOffset=0):
#     if type(src) != np.ndarray:
#         src = ana(src)
#     dst = np.array([[0, 0], [0, dstHeight-1], [dstWidth-1, dstHeight-1], [dstWidth-1,0]])
#     H = find_homography3D(src, dst, zOffset)
#     H_inv = np.linalg.inv(H)
#     boundBox = [imageWidth, imageHeight]

#     xd, yd = np.meshgrid(np.arange(boundBox[0]), np.arange(boundBox[1]))
#     coordinateMatrix = np.stack((yd, xd), axis=-1)
#     output = np.zeros((dstHeight, dstWidth, 2), dtype=int)
#     for i in range(output.shape[1]):
#         for j in range(output.shape[0]):
#             x, y, z, w = H_inv @ ana([i, j, 0.1, 1])
#             x /= w/z
#             y /= w/z
#             print(x,y,z)
#             if 0 <= x and x < boundBox[0] and 0 <= y and y < boundBox[1]:
#                 output[j, i] = coordinateMatrix[int(y), int(x)]
#     return output[:,:,0], output[:,:,1]

def find_homography(src_pts, dst_pts):
    '''STANDARD'''
    A = []
    for i in range(4):
        x, y = src_pts[i][0], src_pts[i][1]
        u, v = dst_pts[i][0], dst_pts[i][1]
        A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
    
    A = np.array(A)
    # Solve Ah = 0 using SVD
    U, S, Vh = np.linalg.svd(A)
    H = Vh[-1,:].reshape(3,3)

    return H / H[2,2]  # Normalize so that h33 = 1

# def find_homography3D(src_pts, dst_pts, zOffset):
#     A = []
#     for i in range(4):
#         x, y = src_pts[i][0], src_pts[i][1]
#         u, v = dst_pts[i][0], dst_pts[i][1]
#         z, w = 0, 0
#         A.append([-x, -y, -z, -1, 0, 0, 0, 0, 0, 0, 0, 0, x*u, y*u, z*u, u])
#         A.append([0, 0, 0, 0, -x, -y, -z, -1, 0, 0, 0, 0, x*v, y*v, z*v, v])
#         A.append([0, 0, 0, 0, 0, 0, 0, 0,-x, -y, -z, -1, x*w, y*w, z*w, w])
    
#     A = ana(A)
#     # # Solve Ah = 0 using SVD
#     U, S, Vh = np.linalg.svd(A)
#     H = Vh[-1,:].reshape(4,4)

#     return H / H[3,3]  # Normalize so that h44 = 1

def getFinalTransform(yw,xw,yu,xu):
    '''MY OWN'''
    temp = np.stack((yu, xu), axis=-1)
    temp = temp[yw, xw]
    y,x = temp[:,:,0],temp[:,:,1]
    return y,x

def FFT(x: np.ndarray):
    '''MY OWN'''
    N = x.shape[0]
    if np.log2(N) % 1 > 0:
        raise ValueError('Must be a power of 2')
    N_min = min(N, 2)
    n = np.arange(N_min)
    k = n[:, np.newaxis]
    M = np.exp(-2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))
    while X.shape[0] < N:
        X_even = X[:, :int(X.shape[1]//2)]
        X_odd = X[:, int(X.shape[1]//2):]
        terms = np.exp(-1j * np.pi * np.arange(X.shape[0])/X.shape[0])[:, np.newaxis]
        X = np.vstack([X_even + terms * X_odd, X_even - terms * X_odd])
    return X.ravel()

def rollRight(array: np.ndarray, n=1):
    '''INSIGNIFICANT'''
    array = list(array)
    return array[-n:] + array[:-n]

def overlayImage(bottom: np.ndarray, top:np.ndarray, rot=0, centre=-1):
    rot = np.deg2rad(rot)
    bot = bottom.copy()
    if centre == -1:
        centre = (bot.shape[0]//2, bot.shape[1]//2)
    alpha = False
    if top.shape[2] == 4:
        alpha = True
    t = []
    k = []
    for x in range(bot.shape[1]):
        x -= centre[1]
        for y in range(bot.shape[0]):
            y -= centre[0]
            r = (x**2 + y**2)**0.5
            theta = np.atan2(y,x)
            if r < np.abs(top.shape[1]/(2*np.cos(theta + rot))) and r < np.abs(top.shape[0]/(2*np.sin(theta + rot))):
                oldX = r*np.cos(theta + rot) + top.shape[1]//2
                oldY = r*np.sin(theta + rot) + top.shape[0]//2
                if alpha:
                    colour = bilinear_interpolation(top, oldX, oldY)
                    bot[y + centre[0], x + centre[1]] = (colour[:3]*(colour[3]/255)) + (bot[y + centre[0], x + centre[1]] * (1 - colour[3]/255))

    # bot[yDest, xDest] = [255,0,0]


    return bot

def rotatePoints(xs,ys,phi):
    x = xs.copy()
    y = ys.copy()
    r = (x**2 + y**2)**0.5
    theta = np.atan2(y,x)
    x = r*np.cos(theta - phi)
    y = r*np.sin(theta - phi)
    return x,y


# def overlayImage(bottom: np.ndarray, top:np.ndarray, rot=0, centre=-1):
#     rot = np.deg2rad(rot)
#     bot = bottom.copy()
#     if centre == -1:
#         centre = (bot.shape[0]//2, bot.shape[1]//2)
#     alpha = False
#     if top.shape[2] == 4:
#         alpha = True

#     centreTop = (top.shape[0]//2, top.shape[1]//2)
#     x,y = np.meshgrid(np.arange(bot.shape[1])-centre[1], np.arange(bot.shape[0])-centre[0])
#     theta = np.atan2(y,x)
#     r = (x**2 + y**2)**0.5
#     mask = np.where((r < np.abs(top.shape[1]/(2*np.cos(theta + rot)))) & (r < np.abs(top.shape[0]/(2*np.sin(theta + rot)))))
#     y,x = mask
#     y -= centre[0]
#     x -= centre[1]
#     oldX,oldY = rotatePoints(x,y,-rot)
#     y += centre[0]
#     x += centre[1]
#     oldX = oldX.astype(int) + centreTop[1]
#     oldY = oldY.astype(int) + centreTop[0]
#     alpha = top[...,3] > 0
#     top = top[...,:3]
#     top[alpha] = [255,0,0]
#     bot[y,x] = top[oldY, oldX,:3]
#     return bot
    

def convolveMultiplication(frame: np.ndarray, kernel: np.ndarray, mode='full'):
    '''MY OWN'''
    if type(frame) == list:
        frame = np.array(frame)
    if type(kernel) == list:
        kernel = np.array(kernel)
    frame = frame.copy()
    kernel = kernel.copy()
    if kernel.size < frame.size:
        frame, kernel = kernel, frame

    
    m1,n1 = frame.shape
    m2,n2 = kernel.shape
    paddedKernel = np.zeros((m1+m2-1, n1+n2-1))
    frameVector = (frame[::-1][:]).reshape(-1)[:,np.newaxis]
    paddedKernel[-m2:, :n2] = kernel
    toeplitz = []
    for F in paddedKernel:
        temp = []
        for i in range(n1):
            temp.append(rollRight(F, i))
        toeplitz.append(ana(temp).T)
    toeplitz = toeplitz[::-1]
    doublyBlocked = []
    for i in range(m1):
        doublyBlocked.append(np.vstack(rollRight(toeplitz,i)))
    doublyBlocked = np.hstack(doublyBlocked)
    output = (doublyBlocked @ frameVector).reshape(m1+m2-1,n1+n2-1)[::-1][:]
    return output

def positive(array):
    '''MY OWN'''
    array = array.copy()
    condition = array < 0
    array[condition] = 0
    return array

# def IFFT(X: np.ndarray):
#     '''MY OWN'''
#     N = X.shape[0]
#     if np.log2(N) % 1 > 0:
#         raise ValueError('Must be a power of 2')
#     N_min = min(N, 2)
#     n = np.arange(N_min)
#     k = n[:, np.newaxis]
#     M = np.exp(2j * np.pi * n * k / N_min)
#     X = np.dot(M, x.reshape((N_min, -1)))
#     while X.shape[0] < N:
#         X_even = X[:, :int(X.shape[1]//2)]
#         X_odd = X[:, int(X.shape[1]//2):]
#         terms = np.exp(1j * np.pi * np.arange(X.shape[0])/X.shape[0])[:, np.newaxis]
#         X = np.vstack([X_even + terms * X_odd, X_even - terms * X_odd])
#     return X.ravel()
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=cameraWidth,
    capture_height=cameraHeight,
    display_width=cameraWidth,
    display_height=cameraHeight,
    framerate=1,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )