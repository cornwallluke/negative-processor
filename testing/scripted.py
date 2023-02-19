# %%
import numpy as np

import cv2

from scipy.ndimage import convolve

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["figure.figsize"] = 10, 15

# %%
import glob

raws = []
ifrraws = []

offset = 0


ibp = np.asarray([float('inf') for i in range(3)])
iwp = np.zeros((3))
# for i in sorted(glob.glob("../photos/positives/France 1976/*.NEF")):
for i in sorted(glob.glob("../photos/misc old/normal/*.tiff"))[offset:]:
    # print(i)
    # raw = rawpy.imread(i)
    # # rpyraws.append(raw)
    pog = cv2.imread(i)

    pog = cv2.cvtColor(pog, cv2.COLOR_BGR2RGB)
    # # imcopy = raw.raw_image_visible.copy()
    # # pog = np.concatenate((
    # #     convolve((imcopy*(raw.raw_colors_visible==0)), H_RB)[:,:,np.newaxis],
    # #     convolve((imcopy*(raw.raw_colors_visible%2)), H_G)[:,:,np.newaxis],
    # #     convolve((imcopy*(raw.raw_colors_visible==2)), H_RB)[:,:,np.newaxis],
    # # ), axis=2)
    # # pog = raw.postprocess(output_bps=16, user_wb=[1, 1, 1, 1], gamma=(1, 1), no_auto_bright=True, user_flip=0)
    # # pog = raw.postprocess(output_bps=16, gamma=(1, 1), user_flip=0)
    # pog = raw.postprocess(output_bps=16, user_wb=[1, 1, 1, 1], gamma=(1, 1), no_auto_bright=True, user_flip=0)
    pog = pog/pog.max()
    ibp = np.minimum(np.percentile(pog[300:-300:10, 300:-300:10], 0.01, axis=(0, 1)), ibp)
    iwp = np.maximum(np.percentile(pog[300:-300:10, 300:-300:10], 99.9, axis=(0, 1)), iwp)
    # pog = pog/pog.max()
    # print(pog.shape)
    raws.append(pog)

for i in sorted(glob.glob("../photos/misc old/infra/*.tiff"))[offset:]:
    # print(i)
    # raw = rawpy.imread(i)
    # # rpyraws.append(raw)
    pog = cv2.imread(i)

    pog = cv2.cvtColor(pog, cv2.COLOR_BGR2RGB)
    # # imcopy = raw.raw_image_visible.copy()
    # # pog = np.concatenate((
    # #     convolve((imcopy*(raw.raw_colors_visible==0)), H_RB)[:,:,np.newaxis],
    # #     convolve((imcopy*(raw.raw_colors_visible%2)), H_G)[:,:,np.newaxis],
    # #     convolve((imcopy*(raw.raw_colors_visible==2)), H_RB)[:,:,np.newaxis],
    # # ), axis=2)
    # # pog = raw.postprocess(output_bps=16, user_wb=[1, 1, 1, 1], gamma=(1, 1), no_auto_bright=True, user_flip=0)
    # # pog = raw.postprocess(output_bps=16, gamma=(1, 1), user_flip=0)
    # pog = raw.postprocess(output_bps=16, user_wb=[1, 1, 1, 1], gamma=(1, 1), no_auto_bright=True, user_flip=0)
    pog = pog/pog.max()
    # ibp = np.minimum(np.percentile(pog[300:-300:10, 300:-300:10], 0.01, axis=(0, 1)), ibp)
    # iwp = np.maximum(np.percentile(pog[300:-300:10, 300:-300:10], 99.9, axis=(0, 1)), iwp)
    # pog = pog/pog.max()
    # print(pog.shape)
    ifrraws.append(pog)
raws=np.asarray(raws)

ifrraws=np.asarray(ifrraws)

print(raws.size)

# %%
# plt.imshow(n[0][0].permute(1, 2, 0))
plt.figure(figsize=(3,3))


# pogawbb = np.percentile(raws, 1, (0, 1, 2))
# pogawbw = np.percentile(raws, 99, (0, 1, 2))
# print("wbs calc")
number = len(raws)
outputs = []


gridsize = int(number**0.5//1+(number**0.5%1>0)*1)

for i in range(len(raws))[:number]:

    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.subplot(gridsize, gridsize,i+1)
    # pog = rawsgamma[i + offset] 
    
    # pogmed = np.median(pog, (0, 1))
    # print(i)
    # toshow = (pog[::,::]-pogawbb)/(pogawbw-pogawbb)
    # print(toshow.max(), toshow.min())

    wbd = np.clip((raws[i]), 0, 1)
    # wbd = wbds[i]
    low = np.percentile(wbd[300:-300:10, 300:-300:10], 0.001, axis=(0, 1))
    high = np.percentile(wbd[300:-300:10, 300:-300:10], 99.999, axis=(0, 1))
    # high  = ibp 
    # print(high)
    # print(low)
    # low = ibp * [1, 1, 1]
    # high = iwp
    high, low = low, high
    print(high)
    print(low)
    # toshow = wbd
    toshow = ( wbd -  low)/(high - low)
    toshow = np.clip(toshow, 0,1)**(1/2.2)
    tmp = (toshow * 255).astype(np.uint8)
    # cv2.imwrite("outputfile"+str(i)+".jpg", cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR))
    outputs.append(toshow)
    plt.imshow(toshow*[1, 1, 1])

# %%
plt.figure(figsize=(10,10))


offset = 5
outputs = []

plt.subplot(2, 1,1)
plt.xticks([])
plt.yticks([])
plt.grid(False)

wbd = np.clip((raws[offset]), 0, 1)
# wbd = wbds[i]
low = np.percentile(wbd[300:-300:10, 300:-300:10], 0.001, axis=(0, 1))
high = np.percentile(wbd[300:-300:10, 300:-300:10], 99.999, axis=(0, 1))
high, low = low, high
    # toshow = wbd
toshow = ( wbd -  low)/(high - low)
toshow = np.clip(toshow, 0,1)**(1/2.2)
tmp = (toshow * 255).astype(np.uint8)
# cv2.imwrite("outputfile"+str(i)+".jpg", cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR))
outputs.append(toshow)
plt.imshow(toshow*[1, 1, 1])

plt.subplot(2, 1,2)
plt.xticks([])
plt.yticks([])
plt.grid(False)

wbd = np.clip((ifrraws[offset]), 0, 1)

wbd = np.add.reduce(wbd, axis = 2 ) / 3

print(wbd.min(), wbd.max())

# # low = np.percentile(wbd[300:-300:4, 300:-300:4], 1, axis=(0, 1))
# high = np.percentile(wbd[300:-300:4, 300:-300:4], 1.1, axis=(0, 1))
    # toshow = wbd

# print(low, high)
# toshow = ( wbd -  low)/(high - low)
toshow = np.clip(wbd, 0, 1)
tmp = (toshow * 255).astype(np.uint8)
# cv2.imwrite("outputfile"+str(i)+".jpg", cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR))
outputs.append(toshow)
plt.imshow(toshow, cmap="Greys")



# %%
originalmask = 1-cv2.resize(outputs[1], outputs[0].shape[1::-1])
print(originalmask.min(), originalmask.max(), originalmask.shape)


originalmask = (originalmask > np.percentile(originalmask[400: -400, 400: -400], 99.7)) * np.ones(originalmask.shape)

sqsize = 5
mask = convolve(originalmask, np.ones((sqsize, sqsize)))
print(mask.min(), mask.max())
mask = np.clip(mask, 0, 1)
print(mask.min(), mask.max())

sqsize = 51
gaussmask = cv2.GaussianBlur(originalmask, (sqsize, sqsize), 0)

sqsize = 19
largeconvmask = convolve(originalmask, np.ones((sqsize, sqsize)))
largeconvmask = np.clip(largeconvmask, 0, 1)




plt.figure(figsize=(10,10))

plt.subplot(1, 1,1)
plt.xticks([])
plt.yticks([])
plt.grid(False)

plt.imshow(largeconvmask)



# %%
raw = raws[offset]

rawgrey = np.average(raw, 2)# np.average.reduce(raw, 2)
threshs = np.percentile(rawgrey[400: -400, 400: -400], np.asarray(range(0, 16), dtype=np.float32) * 0.05)

threshs = sorted(set(threshs))
plt.figure(figsize=(100,100))


# pogawbb = np.percentile(raws, 1, (0, 1, 2))
# pogawbw = np.percentile(raws, 99, (0, 1, 2))
# print("wbs calc")
number = len(raws)
threshoutputs = []


gridsize = 4

for i in threshs:
    threshoutputs.append((rawgrey > i) * np.ones(rawgrey.shape))

# print("drawing")
# for i in range(len(threshs)):

#     plt.subplot(gridsize, gridsize, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)

#     plt.imshow((1 - threshoutputs[i]))

# %%
raw = raws[offset]

rawgrey = np.average(raw, 2) #np.minimum.reduce(raw, 2)

threshs = np.percentile(rawgrey[400: -400, 400: -400], np.asarray(range(0, 16), dtype=np.float32) * 0.05)

print(sorted(set(threshs)))
# pogawbb = np.percentile(raws, 1, (0, 1, 2))
# pogawbw = np.percentile(raws, 99, (0, 1, 2))
# print("wbs calc")
number = len(raws)
threshscores = []


gridsize = 4

idealThresh = None
hiscore = 0



for i in range(len(threshs)):
    threshd = (rawgrey > threshs[i]) * np.ones(rawgrey.shape)
    masked = (1 - threshd) * largeconvmask
    unmasked = (1 - threshd) * (1 - largeconvmask)
    a = 0.8
    # (np.sum(unmasked) * a + (1 - a) * np.sum(largeconvmask)
    positive = np.sum(masked)
    score = positive - (np.sum(unmasked) * a + (1 - a) * np.sum(1 - threshd))
    threshscores.append(score)

    if score >= hiscore:
        hiscore = score
        idealThresh = threshs[i]

    if score < hiscore:
        break


    print(i, score, positive, np.sum(unmasked), np.sum(1 - threshd))

# %%
rawtoalign = (rawgrey[400: -400, 400: -400] > idealThresh) * np.ones(tuple(np.asarray(rawgrey.shape) - [800, 800]))

alignmask = gaussmask[400: -400, 400: -400]

offsets = np.asarray([1, 0])


hishift = None
hiscore = 1
phs = 0

limit = 300

workingshift = [0, 0]

while hiscore > phs and limit > 0:
    limit -= 1
    phs = hiscore
    hiscore = 0
    for i in range(4):
        thet = i * np.pi * 2 / 4
        shift = np.matmul(np.asarray([[np.cos(thet), -np.sin(thet)], [np.sin(thet), np.cos(thet)]]), offsets)
        shift = np.ndarray.astype(shift, np.int8) + workingshift

        # tmat = np.roll(np.asarray(range(16)).reshape((4,4)), offset, (0, 1))
        # # tmat[offset[0] if offset[0] < 0 else None : offset[0] if offset[0] > 0 else None] = 0
        # tmat[:, offset[1] if offset[1] < 0 else None : offset[1] if offset[1] > 0 else None] = 0
        # print(tmat)

        shiftedmask = np.roll(alignmask, shift, (0, 1))

        if shift[0] != 0:
            shiftedmask[shift[0] if shift[0] < 0 else None : shift[0] if shift[0] > 0 else None] = 0
        if shift[1] != 0:
            shiftedmask[:, shift[1] if shift[1] < 0 else None : shift[1] if shift[1] > 0 else None] = 0
        

        score = np.sum(shiftedmask * (1 - rawtoalign))

        if score > hiscore and score > phs:
            hiscore = score
            hishift = shift
    
    print(hiscore, limit, hishift)
    workingshift = hishift
    
offsetmask = np.roll(mask, hishift, (0, 1))

    



# %%
# plt.figure(figsize=(40,40))

# plt.subplot(1, 1,1)
# plt.xticks([])
# plt.yticks([])
# plt.grid(False)

# offsetmask = np.roll(mask, hishift, (0, 1))

# plt.imshow(outputs[0] * ( (1 - np.repeat(offsetmask[:, :, np.newaxis], 3, axis=2)) * [1,0,0] + [0, 1, 1]))

# %%
# plt.figure(figsize=(30,30))

# plt.subplot(1, 1,1)
# plt.xticks([])
# plt.yticks([])
# plt.grid(False)

# plt.imshow(outputs[0])

# %%
plt.figure(figsize=(10,10))

plt.subplot(1, 1,1)
plt.xticks([])
plt.yticks([])
plt.grid(False)
typedreal = (outputs[0] * 2 ** 8).astype(np.uint8)



typedmask = (offsetmask).astype(np.uint8)

print(typedmask.min(), typedmask.max())

dst = cv2.inpaint(typedreal, typedmask, 3, cv2.INPAINT_TELEA)

plt.imshow(dst)

# %%
plt.figure(figsize=(10,1))

plt.subplot(2, 1, 1)
plt.xticks([])
plt.yticks([])
plt.grid(False)

plt.imshow(rawgrey[:300])
plt.subplot(2, 1, 2)
plt.xticks([])
plt.yticks([])
plt.grid(False)

# lsd = cv2.LineSegmentDetector("Refine", "Standard")

# lines = lsd.detect()
plt.imshow(cv2.Canny((cv2.GaussianBlur(rawgrey[:300], (7, 7), 0) * 255).astype(np.uint8), 70, 80))
# lsd.drawSegments(rawgrey[:300:20], lines))

# %%
plt.figure(figsize=(10,2))

plt.subplot(3, 1, 1)
plt.xticks([])
plt.yticks([])
plt.grid(False)

plt.imshow(rawgrey[:300])


plt.subplot(3, 1, 2)
plt.xticks([])
plt.yticks([])
plt.grid(False)

canny = cv2.Canny((cv2.GaussianBlur(rawgrey[:300], (7, 7), 0) * 255).astype(np.uint8), 70, 80)

plt.imshow(canny)

lsd = cv2.createLineSegmentDetector()
print(lsd.detect)

lines = lsd.detect(canny)
plt.imshow(lsd.drawSegments(rawgrey[:300:20], lines))


