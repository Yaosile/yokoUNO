import numpy as np
def rgb2gray(image):
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

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

def scanLines(frame: np.ndarray,x,y,thresholdAmount):
    top = np.mean(frame[:y,:],axis=1)
    bot = np.mean(frame[y:,:],axis=1)
    left = np.mean(frame[:,:x],axis=0)
    right = np.mean(frame[:,x:],axis=0)
    top = largestConsecutiveSet(np.where(top < thresholdAmount))
    bot = largestConsecutiveSet(np.where(bot < thresholdAmount))
    left = largestConsecutiveSet(np.where(left < thresholdAmount))
    right = largestConsecutiveSet(np.where(right < thresholdAmount))
    top = top[-1] if len(top) > 1 else [0]
    bot = bot[0] if len(bot) > 1 else [0]
    left = left[-1] if len(left) > 1 else [0]
    right = right[0] if len(right) > 1 else [0]

    return top,bot+y,left,right+x

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