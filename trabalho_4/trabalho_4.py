import cv2
import matplotlib.pyplot as plt
import numpy as np

def match(imgA,imgB,func):
    
    kp1, des1 = func.detectAndCompute(imgA,None)
    kp2, des2 = func.detectAndCompute(imgB,None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    good = []
    good_without_list = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
            good_without_list.append(m)
    imMatches = cv2.drawMatchesKnn(imgB,kp2,imgA,kp1,good,None,flags=2)

    match = np.asarray(good)
    if len(match[:,0]) >= 4:
        src = np.float32([ kp1[m.queryIdx].pt for m in match[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ kp2[m.trainIdx].pt for m in match[:,0] ]).reshape(-1,1,2)

    H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    print(H)
    dst = cv2.warpPerspective(imgA,H,(imgB.shape[1] + imgA.shape[1], imgB.shape[0]))
    
    dst[0:imgA.shape[0], 0:imgA.shape[1]] = imgB
    return imMatches,dst

def brief_match(imgA,imgB):
    star = cv2.xfeatures2d.StarDetector_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp1 = star.detect(imgA,None)
    kp1, des1 = brief.compute(imgA, kp1)
    kp2 = star.detect(imgA,None)
    kp2, des2 = brief.compute(imgB, kp2)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    good = []
    good_without_list = []
    good = []
    good_without_list = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
            good_without_list.append(m)
    imMatches = cv2.drawMatchesKnn(imgA,kp1,imgB,kp2,matches,None,flags=2)
    #dst = cv2.warpPerspective(imgA,H,(imgB.shape[1] + imgB.shape[1], imgB.shape[0]))
    dst = None
    
    return imMatches,dst
    

imgA = cv2.imread("img/foto1A.jpg")
imgB = cv2.imread("img/foto1B.jpg")

sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()

orb = cv2.ORB_create()
#imMatches,dst =  match(imgA,imgB,orb)
imMatches,dst = brief_match(imgA,imgB)
#cv2.imwrite("matches.jpg", imMatches)
plt.imshow(imMatches)
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()

"""
cv2.imwrite("output.jpg",dst)
plt.imshow(dst)
plt.show()
cv2.
"""
