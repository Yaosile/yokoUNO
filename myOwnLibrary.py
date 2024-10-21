import numpy as np
from numpy import asanyarray as ana
import numba

def normalise(array, minOut = 0, maxOut = 255):
    array = array.copy()
    minIn = np.min(array)
    maxIn = np.max(array)
    array -= minIn
    array /= maxIn
    array *= maxOut
    array += minOut
    return array

def rgb2gray(image):
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

def rgb2hsv(image, Calculations = 'HSV'):
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
    temp = np.zeros(arrayLike.shape)
    threshHeld = np.where(arrayLike > hold)
    temp[threshHeld] = 255
    return temp

def onCardLines(frame: np.ndarray):
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
    valid = np.where(frame > 254)
    t = valid[0][0]
    b = valid[0][-1]
    l = np.min(valid[1])
    r = np.max(valid[1])

    return t,b,l,r

def getRadius(frame, centreX, centreY):
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

def isolateCard(frame):
    t,b,l,r = boundingBox(frame)
    cx,cy = midPoint(t,b,l,r)
    r = getRadius(frame, cx,cy)
    theta = getRotation(frame, cx,cy,r)
    return rotate(frame[-r+cy:r+cy, -r+cx:r+cx], theta)

def getRotation(frame, centreX, centreY, radius):
    radius += 10
    temp = frame[-radius+centreY:radius+centreY, -radius+centreX:radius+centreX]
    xo,yo = np.meshgrid(np.arange(radius*2)-radius, np.arange(radius*2)-radius)
    r,b = (xo**2 + yo**2)**0.5, np.atan2(yo,xo)
    change = np.pi/4
    sumChange = change
    rotated = temp.copy()
    for i in range(50):
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
    midX = (left+right)//2
    midY = (top+bottom)//2
    return midX, midY

def closestValueInSet(value, setOfValues):
    return np.argmin(np.abs(setOfValues - value))

def gaussianKernelGenerator(size, sigma = 2):
    x,y = np.meshgrid(np.arange(size)-size/2+0.5, np.arange(size)-size/2+0.5)
    output = np.exp(-((x**2 + y**2)**0.5)/(2*(sigma**2)))
    output /= output.sum()
    return output

@numba.jit(nopython = True)
def generateGrid(width, height):
    xd,yd = np.zeros_like((height, width)), np.zeros_like((height, width))
    for i in range(width):
        for j in range(height):
            xd[j, i] = i
            yd[j, i] = j
    return xd,yd

@numba.jit(nopython = True)
def distortionMap(distotionCoefficients, mtx, width, height):
    K1 = distotionCoefficients[0]
    K2 = distotionCoefficients[1]
    K3 = distotionCoefficients[4]

    xc = mtx[0][2]
    yc = mtx[1][2]
    fx = mtx[0][0]
    fy = mtx[1][1]
    xd,yd = generateGrid(width, height)

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

def unwarpMap(src, dstWidth, dstHeight, imageWidth, imageHeight):
    if type(src) != np.ndarray:
        src = ana(src)
    dst = np.array([[0, 0], [0, dstHeight-1], [dstWidth-1, dstHeight-1], [dstWidth-1,0]])
    H = find_homography(src, dst)
    H_inv = np.linalg.inv(H)
    boundBox = [imageWidth, imageHeight]

    xd, yd = np.meshgrid(np.arange(boundBox[0]), np.arange(boundBox[1]))
    coordinateMatrix = np.stack((yd, xd), axis=-1)
    output = np.zeros((dstHeight, dstWidth, 2), dtype=int)
    for i in range(output.shape[1]):
        for j in range(output.shape[0]):
            x, y, z = np.dot(H_inv, ana([i, j, 1]))
            x, y = x/z, y/z
            if 0 <= x and x < boundBox[0] and 0 <= y and y < boundBox[1]:
                output[j, i] = coordinateMatrix[int(y), int(x)]
    return output[:,:,0], output[:,:,1]

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
    temp = np.stack((yu, xu), axis=-1)
    temp = temp[yw, xw]
    y,x = temp[:,:,0],temp[:,:,1]
    return y,x

def FFT(x: np.ndarray):
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
    array = list(array)
    return array[-n:] + array[:-n]

def convolveMultiplication(frame: np.ndarray, kernel: np.ndarray, mode='full'):
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
    array = array.copy()
    condition = array < 0
    array[condition] = 0
    return array

def IFFT(X: np.ndarray):
    N = X.shape[0]
    if np.log2(N) % 1 > 0:
        raise ValueError('Must be a power of 2')
    N_min = min(N, 2)
    n = np.arange(N_min)
    k = n[:, np.newaxis]
    M = np.exp(2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))
    while X.shape[0] < N:
        X_even = X[:, :int(X.shape[1]//2)]
        X_odd = X[:, int(X.shape[1]//2):]
        terms = np.exp(1j * np.pi * np.arange(X.shape[0])/X.shape[0])[:, np.newaxis]
        X = np.vstack([X_even + terms * X_odd, X_even - terms * X_odd])
    return X.ravel()