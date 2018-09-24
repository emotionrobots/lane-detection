import cv2
import numpy as np
import glob
from moviepy.editor import VideoFileClip,ImageSequenceClip
import natsort
# import os
# i=1
# for filename in os.listdir("."):
#     if filename.startswith("temp_"):
#         os.rename(filename, str(i)+".png")
#         i=i+1
img=[]
imgfilename=[i for i in glob.glob("*.png")]
    #imgfilename.append(i)
    

imgfilename=natsort.natsorted(imgfilename)
for i in imgfilename:
    imgfile=cv2.imread(i)
    #imgfile=cv2.cvtColor(imgfile,cv2.COLOR_BGR2RGB)
    img.append(imgfile)

height,width,layers=img[1].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter('video.mp4', fourcc, 10.0, (width, height))

for i in img:
    out.write(i)
    cv2.imshow("sg",i)

#video=cv2.VideoWriter('video.mp4',-1,1,(width,height))
#clip = ImageSequenceClip(img, fps=15)
#clip.write_videofile("movie.webm",codec='libvpx')

out.release()
cv2.destroyAllWindows()
#video.release()