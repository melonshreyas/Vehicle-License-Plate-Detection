from imutils import build_montages
from imutils import paths
import numpy as np
import cv2
from matplotlib import pyplot as plt

############## Reading Input Vehicle Image ###########################################

img=cv2.imread('F:\MAJOR PROJECT\code exceution\Preprocessing Step\car15.jpg',1) #Reading an Image
size=(1000,600)
img=cv2.resize(img,size,interpolation=cv2.INTER_AREA)
cv2.imshow('Input Image',img)

###################### Color To Gray scale Image ####################################

row,col,numchan=img.shape
print(row,col,numchan)

imggrayscale=np.zeros((row,col,1),dtype=np.uint8)#intializing to zero
imgthresh=np.zeros((row,col,1),dtype=np.uint8)   #intializing to zero
imgcontours=np.zeros((row,col,3),dtype=np.uint8) #intializing to zero

#Converting BGR i.e. RGB colour to HSV OpenCV use BGR color model instead of RGB
#Advantages of using HSV 
imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
imgHue,imgSaturation,imgValue=cv2.split(imgHSV)

imggrayscale=imgValue
#cv2.imshow('Gray Scale Image',imggrayscale)

rowg,colg=imggrayscale.shape

######################  BILATERAL FILTER  ##########################################

imgBlurred1=cv2.bilateralFilter(imggrayscale,9,13,13)
#cv2.imshow('Bilateral Blurred image',imgBlurred1)

######################  Adaptive Histogram Equalisation #############################

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
imgAHE= clahe.apply(imgBlurred1)
#cv2.imshow('AHE IMAGE',imgAHE)

##################### Morphologically Opening #######################################

#structureelement=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
#imgopened=cv2.morphologyEx(imgAHE,cv2.MORPH_OPEN,structureelement)
kernel = np.ones((51,51),np.uint8)
imgopened = cv2.morphologyEx(imgAHE, cv2.MORPH_OPEN, kernel)
#cv2.imshow('OPENED IMAGE',imgopened)

##################### Difference Image ##############################

differenceimage=cv2.subtract(imgAHE,imgopened)
#cv2.imshow('Difference image ',differenceimage)

##################### Binarized Image ###############################

ret,thresh1 = cv2.threshold(differenceimage,100,255,cv2.THRESH_BINARY)  
#ret,thresh1 = cv2.threshold(differenceimage,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('Threshold Image',thresh1)
'''
##################### Adaptive Threshold - lighting condition variation can be removed ##########

#adaptiveThreshold(src, dst, maxValue, adaptiveMethod, thresholdType, blockSize, C)
#imgThresh=cv2.adaptiveThreshold(differenceimage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,37,20)
#cv2.imshow('Threshold Image',imgThresh)

'''
f=np.fft.fft2(thresh1)
fshift=np.fft.fftshift(f)
p=[]
p=(sum(10*np.log10(abs(fshift)**2))).tolist()
x=[]
for i in range(0,col):
      x.append(i)
img1=np.zeros(thresh1.shape)
M=p.index(max(p))
img1=thresh1[M-(300):M+(50),:]

img4=np.zeros(img.shape)
img4=img[M-(300):M+(50),:]

plt.plot(p)
#plt.imshow(img1)
cv2.imshow('plate',img1)

####################   STEP 2      ###################################

#################### Edge Analysis ###################################
### Using Sobel operator used ####
'''
sobelx1 = cv2.Sobel(thresh1,cv2.CV_64F,1,0,ksize=3)
sobely1 = cv2.Sobel(thresh1,cv2.CV_64F,0,1,ksize=3)

sobelx = np.uint8(np.absolute(sobelx1))
sobely = np.uint8(np.absolute(sobely1))

sobel=(sobelx**2+sobely**2)**0.5
sobel=np.uint8(sobel)
#sobel=cv2.cvtColor(sobel,cv2.COLOR_GRAY2BGR)
ret,sobel = cv2.threshold(sobel,0,1,cv2.THRESH_BINARY)
sobel=cv2.cvtColor(sobel,cv2.COLOR_GRAY2BGR)
plt.imshow(sobel)
'''
######### Canny Edge Anaysis #########################################

sobel = cv2.Canny(img1,100,200)
cv2.imshow('Edges',sobel)

######### Dilation of Image ##########################################

kernel = np.ones((3,3),np.uint8)
dilation = cv2.dilate(sobel,kernel,iterations = 1)
################### Conversion Float64 to int8########################
#dilation =np.uint8(dilation)
#cv2.imshow('Dilated Edge',dilation)

######### Hole Filling ###############################################

# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
im_floodfill = dilation.copy()
#im_floodfill = np.uint8(np.absolute(dilation))
h, w = dilation.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)

# Floodfill from point (0,0)
cv2.floodFill(im_floodfill,mask,(0,0),255)


# Invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(im_floodfill)
 
# Combine the two images to get the foreground.
im_out = dilation | im_floodfill_inv

#cv2.imshow("Floodfilled Image", im_floodfill)
#cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
cv2.imshow("Foreground", im_out) # im_out is having license plate region hole filled region
#####################################################################################
################ opening ############################################################
kernel = np.ones((20,20),np.uint8)
img_opening=cv2.morphologyEx(im_out, cv2.MORPH_OPEN, kernel)
cv2.imshow("Opened Image of the hole filled", img_opening)

######################## Erosion ###################################################
erosion = cv2.erode(img_opening,kernel,iterations = 1)
cv2.imshow("Eroded Image",erosion)


########################## Connected Component Analysis - CCA #######################

connectivity = 4
output=[]
output = cv2.connectedComponentsWithStats(erosion,connectivity, cv2.CV_32S)
img2=np.zeros(erosion.shape)
img3=np.zeros(erosion.shape)
img4=np.zeros(erosion.shape)
stats = output[2]
#print(stats)
href=0;
wref=0;
for i in range(len(stats)):
    m=stats[i]
    x=m[0]
    y=m[1]
    w=m[2]
    h=m[3]
    area=m[4]
    if ((w/h)>2):
        if (area>1000):
            img2 = erosion[y-2:y+h+2,x-2:x+w+2]
            img3 = img1[y:y+h+8,x:x+w+8]
            #cv2.imshow('image2',img2)
            #cv2.imwrite('F:\MAJOR PROJECT\code exceution\Preprocessing Step\obtained.png',img3)
            #size=img2.shape
            #img4=cv2.resize(img4,size,interpolation=cv2.INTER_AREA)
            #img2=cv2.resize(img2,size,interpolation=cv2.INTER_AREA)
            #vis = np.concatenate((img4,img2), axis=1)
            cv2.imwrite('F:\MAJOR PROJECT\code exceution\Preprocessing Step\out.png',img3)

    cv2.imshow('Car Plate Image',img3)
    cv2.imwrite('F:\MAJOR PROJECT\code exceution\Preprocessing Step\carplate11.jpg',img3)
    #plt.imshow(img4) 
    #plt.show()


'''
######################################################################
img=cv2.imread(r'F:\MAJOR PROJECT\code exceution\Preprocessing Step\obtained.png',1)
size=(600,150)
img=cv2.resize(img,size,interpolation=cv2.INTER_AREA)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,thresh_image = cv2.threshold(img,70,255,cv2.THRESH_BINARY_INV)

#cv2.imwrite('F:\MAJOR PROJECT\code exceution\Character Recognition\Carplatebinary6.jpg',thresh_image)

#cv2.imshow('Binary',img) 
#plt.imshow(img) Binary image
#plt.show()
#cv2.waitKey(0)

kernel = np.ones((5,5),np.uint8)
# Opening- erosion followed by dilation to remove noise
#opening = cv2.morphologyEx(thresh_image,cv2.MORPH_OPEN,kernel)
#scv2.imshow('img',opening)

#Closing- erosion followed by dilation to remove noise
closing = cv2.morphologyEx(thresh_image,cv2.MORPH_CLOSE,kernel)
cv2.imshow('img1',closing)
# Opening 
opening1 = cv2.morphologyEx(closing,cv2.MORPH_OPEN,kernel)
cv2.imshow('img2',opening1)


######################################## Connected Component Analysis ###############################

img = np.uint8(opening1)
connectivity = 4
output=[]

# Perform the operation
output = cv2.connectedComponentsWithStats(img,connectivity, cv2.CV_32S)

# The first cell is the number of labels
num_labels = output[0]

# The second cell is the label matrix
labels = output[1]

# The third cell is the stat matrix
stats = output[2]
print(stats)

# The fourth cell is the centroid matrix
centroids = output[3]
ws=[]
hs=[]
################## Draw Rectangle ##########################
for i in range(len(stats)):
    m=stats[i]
    x=m[0]
    y=m[1]
    w=m[2]
    h=m[3]
    area=m[4]
    ws.append(w)
    hs.append(h)
    if (w>5 and w<65):
        if (h>25 and h<120):
            if (area>620 and area<3200):
                img1 = cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),1)
                img1=img[y-2:y+h+2,x-2:x+w+2]
                plt.imshow(img1)
                plt.show()
'''
plt.show()
cv2.waitKey(0)
