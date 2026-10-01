from __future__ import division
import os
import cv2
import torch

def numpy2cv2(cont,style,prop,width,height):
    cont = cont.transpose((1,2,0))
    cont = cont[...,::-1]
    cont = cont * 255
    cont = cv2.resize(cont,(width,height))
    #cv2.resize(iimg,(width,height))
    style = style.transpose((1,2,0))
    style = style[...,::-1]
    style = style * 255
    style = cv2.resize(style,(width,height))

    prop = prop.transpose((1,2,0))
    prop = prop[...,::-1]
    prop = prop * 255
    prop = cv2.resize(prop,(width,height))

    #return np.concatenate((cont,np.concatenate((style,prop),axis=1)),axis=1)
    return prop,cont

def makeVideo(content,style,props,outf):
    print('Stack transferred frames back to video...')
    layers,height,width = content[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(os.path.join(outf,'transfer.mp4'),fourcc,10.0,(width,height))
    ori_video = cv2.VideoWriter(os.path.join(outf,'content.mp4'),fourcc,10.0,(width,height))
    for j in range(len(content)):
        prop,cont = numpy2cv2(content[j],style,props[j],width,height)
        cv2.imwrite('prop.png',prop)
        cv2.imwrite('content.png',cont)

        imgj = cv2.imread('prop.png')
        imgc = cv2.imread('content.png')

        video.write(imgj)
        ori_video.write(imgc)
        # RGB or BRG, yuks
    video.release()
    ori_video.release()
    os.remove('prop.png')
    os.remove('content.png')
    print('Transferred video saved at %s.'%outf)
