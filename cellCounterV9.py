import cv2, json, numpy as np, scipy.misc as sp, sys, random
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage.morphology import distance_transform_edt as dte
from multiprocessing import Pool

class imageProcessor():
    def applyMask(self,imgIn,maskIn):
        imgIn = imgIn.astype(float)
        imgIn[maskIn != 255.] = 0
        return imgIn

    def bufferEdges(self,imgIn,yBuf=0,xBuf=0):
        imgBuff = np.full((imgIn.shape[0]+2*yBuf,imgIn.shape[1]+2*xBuf),0.0)
        imgBuff[yBuf:-yBuf,xBuf:-xBuf]=imgIn
        return imgBuff

    def unbufferEdges(self,imgIn,yBuf=0,xBuf=0):
        return imgIn[yBuf:-yBuf,xBuf:-xBuf]

    def floorBackgroundSubtraction(self,imgIn,minVal=0,replaceVal=0):
        imgIn[imgIn<minVal] = minVal
        imgIn = imgIn - minVal
        return imgIn

    def morphBluring(self,imgIn,kernelSize=5,repeats=1):
        kern = np.ones((kernelSize,kernelSize),np.uint8)
        for i in range(0,repeats):
            imgIn = cv2.GaussianBlur(imgIn,(kernelSize,kernelSize),0)
        return imgIn

    def morphErode(self,imgIn,kernelSize=5,repeats=1):
        kern = np.ones((kernelSize,kernelSize),np.uint8)
        for i in range(0,repeats):
            imgIn = cv2.erode(imgIn,kern,iterations = 1)
        return imgIn

    def xyLocalMaxFinder(self,imgIn,minpeakSpacing=0,minPeakVal=0,negationList=[]):
        ypeaks = []
        for y in range(0,imgIn.shape[0]):
            strip = imgIn[y]
            maxs=find_peaks(strip)[0].tolist()
            ypeaks=ypeaks+[(y,m) for m in maxs]

        xpeaks = []
        for x in range(0,imgIn.shape[1]):
            strip = np.transpose(imgIn)[x]
            maxs=find_peaks(strip)[0].tolist()
            xpeaks=xpeaks+[(m,x) for m in maxs]

        cpeaks = []
        for p in ypeaks:
            if p in xpeaks and imgIn[p[0],p[1]] >= minPeakVal: cpeaks.append(p)

        if len(negationList)!=0:
            cpeaks2=cpeaks[:]
            for n in negationList:
                for c in cpeaks:
                    distance = np.sqrt(abs(n[0]-c[0])**2+abs(n[1]-c[1])**2)
                    if distance<= minpeakSpacing and c in cpeaks2:
                        cpeaks2.remove(c)
            cpeaks=cpeaks2

        cpeaksCleanup = [(c,True) for c in cpeaks]
        for i in range(0,len(cpeaksCleanup)):
            for j in range (0,len(cpeaksCleanup)):
                c1 = cpeaksCleanup[i]
                c2 = cpeaksCleanup[j]
                if i != j and c1[1]:
                    distance = np.sqrt(abs(c1[0][0]-c2[0][0])**2+abs(c1[0][1]-c2[0][1])**2)
                    if distance <=minpeakSpacing: cpeaksCleanup[j] = (cpeaksCleanup[j][0],False)
        cpeaks = [c[0] for c in cpeaksCleanup if c[1]]

        return cpeaks

def onclick(event):
    global ixBG,iyBG,ixSig,iySig
    if event.button == 1:
        ixBG, iyBG = int(round(event.xdata)), int(round(event.ydata))
        plt.plot(ixBG, iyBG, 'ro')
    elif event.button == 3:
        ixSig, iySig = int(round(event.xdata)), int(round(event.ydata))
        plt.plot(ixSig, iySig, 'bo')
    plt.draw()

def poolCore(inAr):
    ip = imageProcessor()
    imgSub=inAr[0]

    mcdp=inAr[1]
    mcs=inAr[2]
    ct=inAr[3]

    lav=inAr[4]
    mcdp2=inAr[5]
    mosrr=inAr[6]

    pf1=inAr[7][0]
    pf2=inAr[7][1]
    pf3=inAr[7][2]

    ppbfr=inAr[8][0]
    ppbfd=inAr[8][1]

    ppbd=inAr[9][0]
    ppdr=inAr[9][1]

    pped=inAr[10][0]
    pper=inAr[10][1]

    atw=inAr[11]
    mcdp3=inAr[12]

    y0,y1,x0,x1,yES,xES=inAr[13][0],inAr[13][1],inAr[13][2],inAr[13][3],inAr[13][4],inAr[13][5]
    sys.stdout.write("\r"+str(ct)+"       ")
    sys.stdout.flush()
    # print np.mean(imgSub)

    imggrab = imgSub

    imgSub = ip.floorBackgroundSubtraction(imgSub,minVal=lav)
    openingKern = np.ones((mcdp2,mcdp2),np.uint8)
    imgSub = cv2.morphologyEx(imgSub, cv2.MORPH_OPEN, openingKern)
    imgSub = ip.floorBackgroundSubtraction(imgSub,minVal=np.mean(imgSub)*mosrr)

    # fig = plt.subplot2grid((2, 2), (0, 0))
    # plt.imshow(imgSub,cmap='gray')

    if pf1: imgSub = cv2.bilateralFilter(np.float32(imgSub),ppbfr,ppbfd,ppbfd)
    if pf2: imgSub = ip.morphBluring(imgSub,kernelSize=ppbd,repeats=ppdr)
    if pf3: imgSub = ip.morphErode(imgSub,kernelSize=pped,repeats=pper)

    imgSub = np.array(imgSub.astype(int), dtype = np.uint8)
    imgSub = cv2.adaptiveThreshold(imgSub,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,atw,0)
    imgSub = cv2.morphologyEx(imgSub, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))

    # plt.subplot2grid((2, 2), (0, 1),sharex=fig,sharey=fig)
    # plt.imshow(imgSub,cmap='gray')

    imgSub = dte(imgSub!=0)
    imgSub = dte((imgSub<=mcdp3)*(imgSub!=0))

    peakList = []
    for mcd in mcdp:
        peakList = peakList + ip.xyLocalMaxFinder(imgSub,minpeakSpacing=mcs,minPeakVal=mcd,negationList=peakList)

    # plt.subplot2grid((2, 2), (1, 0),sharex=fig,sharey=fig)
    # plt.imshow(imgSub,cmap='jet')
    # plt.plot([c[1] for c in peakList],[c[0] for c in peakList],'mo',ms=4)

    # plt.subplot2grid((2, 2), (1, 1),sharex=fig,sharey=fig)
    # plt.imshow(imggrab,cmap='gray')
    # plt.plot([c[1] for c in peakList],[c[0] for c in peakList],'ro',ms=4)

    # plt.show()

    # plt.pause(2.0)
    # plt.clf()

    y0R,y1R = y0+yES,y1-yES
    x0R,x1R = x0+xES,x1-xES
    peakList = [(y+y0,x+x0) for (y,x) in peakList if (y+y0>=y0R and y+y0<=y1R and x+x0>=x0R and x+x0<=x1R)]

    return peakList

if __name__ == "__main__":

    #####################################
    # windowSizeY = 1000
    # windowSizeX = 1000
    windowSizeY = 450
    windowSizeX = 450
    procRatioUsed = 0.8

    # lowestAllowedVal = 22
    # lowestAllowedVal = 13
    # lowestAllowedVal = 9
    # lowestAllowedVal = 15
    # lowestAllowedVal = 'manual'
    lowestAllowedVal = 11

    minObjSizeRemovalRatio = 0.25

    prePBilFil = True
    preProcessingBilFilRepeats = 5
    preProcessingBilFilDimension = 121

    prePBlur = True
    preProcessingBlurringRepeats = 5
    preProcessingBlurringDimension = 5

    preErode = True
    preProcessingErodeRepeats = 1
    preProcessingErodeDimension = 5

    adaptiveThresholdWindow = 121

    minCellDiameterPre = 15
    # minCellDiameterPost = [22,18,14,10,7]
    minCellDiameterPost = [22,18,14,10]
    maxCellDiameter = 50
    # minCellSpacing = 41
    minCellSpacing = 37
    #####################################

    #imgName,maskName,maskFullName = sys.argv[1],sys.argv[2],sys.argv[3]
    imgName = sys.argv[1]
    maskName = imgName+"_full_mask"
    maskFullName = imgName+"_retina_mask"
    imgName,maskName,maskFullName = imgName+".tif",maskName+".tif",maskFullName+".tif"

    img = cv2.imread(imgName,0)#[8000:10000,8000:10000]
    mask = cv2.imread(maskName,0)#[8000:10000,8000:10000]
    maskFull = cv2.imread(maskFullName,0)#[8000:10000,8000:10000]

    # plt.imshow(mask,cmap='gray')
    # plt.show()

    if mask[0,0]==255.:
        mask=(mask==0.)*255.
    if maskFull[0,0]==255.:
        maskFull=(maskFull==0.)*255.

    maskArea = int(round(np.sum(mask/255)))
    maskFullArea = int(round(np.sum(maskFull/255)))

    imgpre = np.copy(img)
    ip = imageProcessor()

    img = ip.applyMask(img,mask)

    plt.imshow(img,cmap='gray')
    plt.show()

    #print img.shape, np.mean(img[img!=0]), np.max(img[img!=0]), np.min(img[img!=0])

    #imgBR = ip.floorBackgroundSubtraction(img,minVal=lowestAllowedVal)

    # fig = plt.subplot2grid((1, 2), (0, 0))
    # plt.imshow(img,cmap='gray')
    # plt.subplot2grid((1, 2), (0, 1),sharex=fig,sharey=fig)
    # plt.imshow(imgBR,cmap='gray')
    # plt.show()

    global iyBG, ixBG, iySig, ixSig
    while lowestAllowedVal == 'manual':
        yOf,xOf = random.randint(0,img.shape[0]-250),random.randint(0,img.shape[1]-250)
        iyBG, ixBG, iySig, ixSig = None, None, None, None
        fig = plt.figure()
        plt.imshow(img[yOf:yOf+250,xOf:xOf+250],cmap='gray')
        cid = fig.canvas.mpl_connect('button_press_event',onclick)
        plt.show()
        #print iyBG, ixBG, iySig, ixSig
        if iyBG != None and ixBG != None and iySig != None and ixSig !=None:
            iyBG, ixBG, iySig, ixSig = iyBG+yOf, ixBG+xOf, iySig+yOf, ixSig+xOf
            lowestAllowedVal = int(round((img[iySig,ixSig]-img[iyBG,ixBG])/4+img[iyBG,ixBG]))
            print img[iyBG,ixBG], img[iySig,ixSig], lowestAllowedVal

            plt.imshow(img,cmap='gray')
            plt.imshow(img>=lowestAllowedVal,alpha=0.1,cmap='jet')
            plt.ion()
            plt.show()

            check = raw_input("Good min vals (Yes=y, No=n): ")
            plt.close()
            plt.ioff()
            if check=='y': break
            else: lowestAllowedVal = 'manual'


    edgeSpaceY,edgeSpaceX = int((1-procRatioUsed)/2*windowSizeY),int((1-procRatioUsed)/2*windowSizeX)
    img = ip.bufferEdges(img,yBuf=edgeSpaceY,xBuf=edgeSpaceX)
    imgpre = ip.bufferEdges(imgpre,yBuf=edgeSpaceY,xBuf=edgeSpaceX)

    edgeSizeY,edgeSizeX = (1-procRatioUsed)/2*windowSizeY,(1-procRatioUsed)/2*windowSizeX
    subimages = []

    count=0
    for i in range(0,img.shape[0],int(windowSizeY/(1/procRatioUsed))):
        for j in range(0, img.shape[1],int(windowSizeX/(1/procRatioUsed))):
            count+=1
            y0,y1 = i, np.minimum(i+windowSizeY,img.shape[0]-1)
            x0,x1 = j, np.minimum(j+windowSizeX,img.shape[1]-1)
            imgSub = img[y0:y1,x0:x1]
            subimages.append((imgSub,minCellDiameterPost,minCellSpacing,count,lowestAllowedVal,minCellDiameterPre,minObjSizeRemovalRatio,
                (prePBilFil, prePBlur, preErode),(preProcessingBilFilRepeats,preProcessingBilFilDimension),
                (preProcessingBlurringDimension,preProcessingBlurringRepeats),(preProcessingErodeDimension,preProcessingErodeRepeats),
                adaptiveThresholdWindow,maxCellDiameter,(y0,y1,x0,x1,edgeSizeY,edgeSizeX)))

    print "Count total: ",count

    p=Pool()
    cellLocs = p.map(poolCore,subimages)
    p.close()
    p.join()
    # print cellLocs

    cellLocs = [val for sublist in cellLocs for val in sublist]

    imgpre = ip.unbufferEdges(imgpre,yBuf=edgeSpaceY,xBuf=edgeSpaceY)
    img = ip.unbufferEdges(img,yBuf=edgeSpaceY,xBuf=edgeSpaceY)
    cellLocs = [(c[0]-edgeSpaceY,c[1]-edgeSpaceX) for c in cellLocs]



    adjustedCT = int(round((len(cellLocs)-8.48)/1.002))
    adjustedCT2 = int(round(float(adjustedCT) * (float(maskFullArea)/float(maskArea))))
    print '\n\nCellCount Actual: ',len(cellLocs)
    print 'CellCount Adjusted: ',adjustedCT
    print 'CellCount Adjusted 2: ',adjustedCT2
    print 'Mask Area: ',maskArea
    print 'Full Mask Area: ',maskFullArea



    plt.imshow(imgpre,cmap='gray')
    yPts=[i[0] for i in cellLocs]
    xPts=[i[1] for i in cellLocs]
    plt.plot(xPts,yPts,'ro',ms=0.5)
    fig = plt.gcf()
    fig.set_size_inches(24, 24)
    fig.savefig(imgName[:-4]+'_Output.png', dpi=400)
    plt.show()
    np.save(imgName[:-4]+'_cellLoc.npy',cellLocs)
    file = open(imgName[:-4]+'_sumStats.txt','w')
    file.write('CellCount_Actual:\t'+str(len(cellLocs)))
    file.write('\nCellCount_Adjusted:\t'+str(adjustedCT))
    file.write('\nCellCount_Adjusted2:\t'+str(adjustedCT2))
    file.write('\nMaskArea:\t'+str(maskArea))
    file.write('\nMaskArea2:\t'+str(maskFullArea))
    file.write('\nLowcutoff:\t'+str(lowestAllowedVal))
    file.close()
