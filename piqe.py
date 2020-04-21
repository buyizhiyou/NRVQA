import numpy as np
import cv2
from scipy.special import gamma


def calculate_mscn(dis_image):
    dis_image = dis_image.astype(np.float32)  # 类型转换十分重要
    ux = cv2.GaussianBlur(dis_image, (7, 7), 7/6)
    ux_sq = ux*ux
    sigma = np.sqrt(np.abs(cv2.GaussianBlur(dis_image**2, (7, 7), 7/6)-ux_sq))

    mscn = (dis_image-ux)/(1+sigma)

    return mscn

# Function to segment block edges


def segmentEdge(blockEdge, nSegments, blockSize, windowSize):
    # Segment is defined as a collection of 6 contiguous pixels in a block edge
    segments = np.zeros((nSegments, windowSize))
    for i in range(nSegments):
        segments[i, :] = blockEdge[i:windowSize]
        if(windowSize <= (blockSize+1)):
            windowSize = windowSize+1

    return segments


def noticeDistCriterion(Block, nSegments, blockSize, windowSize, blockImpairedThreshold, N):
    # Top edge of block
    topEdge = Block[0, :]
    segTopEdge = segmentEdge(topEdge, nSegments, blockSize, windowSize)

    # Right side edge of block
    rightSideEdge = Block[:, N-1]
    rightSideEdge = np.transpose(rightSideEdge)
    segRightSideEdge = segmentEdge(
        rightSideEdge, nSegments, blockSize, windowSize)

    # Down side edge of block
    downSideEdge = Block[N-1, :]
    segDownSideEdge = segmentEdge(
        downSideEdge, nSegments, blockSize, windowSize)

    # Left side edge of block
    leftSideEdge = Block[:, 0]
    leftSideEdge = np.transpose(leftSideEdge)
    segLeftSideEdge = segmentEdge(
        leftSideEdge, nSegments, blockSize, windowSize)

    # Compute standard deviation of segments in left, right, top and down side edges of a block
    segTopEdge_stdDev = np.std(segTopEdge, axis=1)
    segRightSideEdge_stdDev = np.std(segRightSideEdge, axis=1)
    segDownSideEdge_stdDev = np.std(segDownSideEdge, axis=1)
    segLeftSideEdge_stdDev = np.std(segLeftSideEdge, axis=1)

    # Check for segment in block exhibits impairedness, if the standard deviation of the segment is less than blockImpairedThreshold.
    blockImpaired = 0
    for segIndex in range(segTopEdge.shape[0]):
        if((segTopEdge_stdDev[segIndex] < blockImpairedThreshold) or
                (segRightSideEdge_stdDev[segIndex] < blockImpairedThreshold) or
                (segDownSideEdge_stdDev[segIndex] < blockImpairedThreshold) or
                (segLeftSideEdge_stdDev[segIndex] < blockImpairedThreshold)):
            blockImpaired = 1
            break

    return blockImpaired


def noiseCriterion(Block, blockSize, blockVar):
    # Compute block standard deviation[h,w,c]=size(I)
    blockSigma = np.sqrt(blockVar)
    # Compute ratio of center and surround standard deviation
    cenSurDev = centerSurDev(Block, blockSize)
    # Relation between center-surround deviation and the block standard deviation
    blockBeta = (abs(blockSigma-cenSurDev))/(max(blockSigma, cenSurDev))

    return blockSigma, blockBeta

# Function to compute center surround Deviation of a block


def centerSurDev(Block, blockSize):
    # block center
    center1 = int((blockSize+1)/2)-1
    center2 = center1+1
    center = np.vstack((Block[:, center1], Block[:, center2]))
    # block surround
    Block = np.delete(Block, center1, axis=1)
    Block = np.delete(Block, center1, axis=1)

    # Compute standard deviation of block center and block surround
    center_std = np.std(center)
    surround_std = np.std(Block)

    # Ratio of center and surround standard deviation
    cenSurDev = (center_std/surround_std)

    # Check for nan's
    # if(isnan(cenSurDev)):
    #     cenSurDev = 0

    return cenSurDev


def piqe(im):
    blockSize = 16  # Considered 16x16 block size for overall analysis
    activityThreshold = 0.1  # Threshold used to identify high spatially prominent blocks
    blockImpairedThreshold = 0.1  # Threshold identify blocks having noticeable artifacts
    windowSize = 6  # Considered segment size in a block edge.
    nSegments = blockSize-windowSize+1  # Number of segments for each block edge
    distBlockScores = 0  # Accumulation of distorted block scores
    NHSA = 0  # Number of high spatial active blocks.

    # pad if size is not divisible by blockSize
    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    originalSize = im.shape
    rows, columns = originalSize
    rowsPad = rows % blockSize
    columnsPad = columns % blockSize
    isPadded = False
    if(rowsPad > 0 or columnsPad > 0):
        if rowsPad > 0:
            rowsPad = blockSize-rowsPad
        if columnsPad > 0:
            columnsPad = blockSize-columnsPad
        isPadded = True
        padSize = [rowsPad, columnsPad]
    im = np.pad(im, ((0, rowsPad), (0, columnsPad)), 'edge')

    # Normalize image to zero mean and ~unit std
    # used circularly-symmetric Gaussian weighting function sampled out
    # to 3 standard deviations.
    imnorm = calculate_mscn(im)

    # Preallocation for masks
    NoticeableArtifactsMask = np.zeros(imnorm.shape)
    NoiseMask = np.zeros(imnorm.shape)
    ActivityMask = np.zeros(imnorm.shape)

    # Start of block by block processing
    total_var = []
    total_bscore = []
    total_ndc = []
    total_nc = []

    BlockScores = []
    for i in np.arange(0, imnorm.shape[0]-1, blockSize):
        for j in np.arange(0, imnorm.shape[1]-1, blockSize):
             # Weights Initialization
            WNDC = 0
            WNC = 0

            # Compute block variance
            Block = imnorm[i:i+blockSize, j:j+blockSize]
            blockVar = np.var(Block)

            if(blockVar > activityThreshold):
                ActivityMask[i:i+blockSize, j:j+blockSize] = 1
                NHSA = NHSA+1

                # Analyze Block for noticeable artifacts
                blockImpaired = noticeDistCriterion(
                    Block, nSegments, blockSize-1, windowSize, blockImpairedThreshold, blockSize)

                if(blockImpaired):
                    WNDC = 1
                    NoticeableArtifactsMask[i:i +
                                            blockSize, j:j+blockSize] = blockVar

                # Analyze Block for guassian noise distortions
                [blockSigma, blockBeta] = noiseCriterion(
                    Block, blockSize-1, blockVar)

                if((blockSigma > 2*blockBeta)):
                    WNC = 1
                    NoiseMask[i:i+blockSize, j:j+blockSize] = blockVar

                # Pooling/ distortion assigment
                # distBlockScores = distBlockScores + \
                #     WNDC*pow(1-blockVar, 2) + WNC*pow(blockVar, 2)

                if WNDC*pow(1-blockVar, 2) + WNC*pow(blockVar, 2) > 0:
                    BlockScores.append(
                        WNDC*pow(1-blockVar, 2) + WNC*pow(blockVar, 2))

                total_var = [total_var, blockVar]
                total_bscore = [total_bscore, WNDC *
                                (1-blockVar) + WNC*(blockVar)]
                total_ndc = [total_ndc, WNDC]
                total_nc = [total_nc, WNC]

    BlockScores = sorted(BlockScores)
    lowSum = sum(BlockScores[:int(0.1*len(BlockScores))])
    Sum = sum(BlockScores)
    Scores = [(s*10*lowSum)/Sum for s in BlockScores]
    C = 1
    Score = ((sum(Scores) + C)/(C + NHSA))*100

    # if input image is padded then remove those portions from ActivityMask,
    # NoticeableArtifactsMask and NoiseMask and ensure that size of these masks
    # are always M-by-N.
    if(isPadded):
        NoticeableArtifactsMask = NoticeableArtifactsMask[0:originalSize[0],
                                                          0:originalSize[1]]
        NoiseMask = NoiseMask[0:originalSize[0], 0:originalSize[1]]
        ActivityMask = ActivityMask[0:originalSize[0], 1:originalSize[1]]

    return Score, NoticeableArtifactsMask, NoiseMask, ActivityMask
