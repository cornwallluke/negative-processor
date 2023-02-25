import numpy as np

import cv2

from scipy.ndimage import convolve

import sklearn.cluster

import glob

from multiprocessing import Pool

import os

raws = []
ifrraws = []



def listfiles(path):
    # for i in sorted(glob.glob("../photos/positives/France 1976/*.NEF")):
    return list(map(
        lambda x: x.split("/")[-1],
        sorted(glob.glob(f"{path}/normal/*.tiff"), key=len)
    ))

def normaliseAndInvert(img):
    wbd = np.clip((img), 0, 1)
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
    toshow = np.clip(toshow, 0,1)#**(1/2.2) gamma??

    return toshow

def fixDustMask(raw, ifrraw):
    originalmask = 1-cv2.resize(ifrraw, raw.shape[1::-1])
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


    rawgrey = np.average(raw, 2)# np.average.reduce(raw, 2)
    
    threshs = np.percentile(rawgrey[400: -400, 400: -400], np.asarray(range(0, 16), dtype=np.float32) * 0.05 + 0.025)

    threshscores = []

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


    return offsetmask.astype(np.uint8)

def findEdges(rawgrey):

    edgepickers = [
        lambda x: (x[:150], (0, 0), 0),
        lambda x: (x[:, -150:], (x.shape[1], 0), 90),
        lambda x: (x[-150:], (x.shape[1], x.shape[0]) , 180),
        lambda x: (x[:, :150], (0, x.shape[0]), 270),
    ]

    plotn = 1

    cumulativelines = []
    cumderivedlines = []
    cumvectors = []

    for edgepicker in edgepickers:
        target = np.rot90(edgepicker(rawgrey)[0], edgepicker(rawgrey)[2] // 90)
        translation = edgepicker(rawgrey)[1]
        rotation = edgepicker(rawgrey)[2]

        thet = rotation / 180 * np.pi


        toapplyon = cv2.GaussianBlur(target, (11, 1), 0)

        nmin, nmax = np.percentile(target, 10), np.percentile(target, 90)
        toapplyon = np.clip((toapplyon - nmin) / (nmax - nmin)  , 0, 1)

        t = 50
        canny = cv2.Canny((toapplyon * 255).astype(np.uint8), t, 4 * t)

        lsd = cv2.createLineSegmentDetector(0)
        lines, widths, precs, nfas = lsd.detect(canny)


        cumulativelines += [
            [
                tuple(
                    [*(np.matmul(np.asarray([[np.cos(thet), -np.sin(thet)], [np.sin(thet), np.cos(thet)]]), np.asarray(vline[:2])) + np.asarray(translation))] +
                    [*(np.matmul(np.asarray([[np.cos(thet), -np.sin(thet)], [np.sin(thet), np.cos(thet)]]), np.asarray(vline[2:])) + np.asarray(translation))]
                )
            ]
            for vline in lines[:, 0]
        ]

        sortedvals = sorted([
            ((x[3] - x[1]) / (x[2] - x[0]), x[1] - (x[3] - x[1]) / (x[2] - x[0]) * x[0], int(((x[2] - x[0]) ** 2 + (x[3] - x[1]) ** 2) ** .5))
            for x in lines[:, 0] if x[2] != x[0]
        ], key=lambda x: x[0])


        ragged = [
            [m, c] * n for (m, c, n) in sortedvals
        ]

        biasedgrads = np.array(
            [i for k in ragged for i in k]
        ).reshape((-1, 1, 2))


        totalmag = len(biasedgrads)

        if totalmag < 2000:
            vectorisedcandidatelines = np.asarray(
                [
                    [
                        [0, 0],
                        [1, 0]
                    ]
                    for cgrad in candidategrads
                ]
            )
        
        
        else:
            filteredgrads = biasedgrads[int(biasedgrads.shape[0] / 3): -int(biasedgrads.shape[0] / 3)]

            k = 2

            kmeans = sklearn.cluster.KMeans(k, n_init=30).fit(filteredgrads[:, 0])

            # print(sorted(kmeans.cluster_centers_, key=lambda x: x[0])[0])

            categorised = [
                np.asarray(sorted(
                    [filteredgrads[j, 0] for j in range(filteredgrads.shape[0]) if kmeans.labels_[j] == i],
                    key=lambda x: abs(kmeans.cluster_centers_[i][0] - x[0]) + abs(kmeans.cluster_centers_[i][1] - x[1])
                ))
                for i in range(k)  
            ]

            filteredcandidates = list(filter(
                lambda x: len(x) > len(categorised[0]) * 0.2,
                sorted(categorised, key=lambda x: -len(x))
            ))

            candidategrads = [
                np.median(points, 0)
                for points in filteredcandidates
            ]

            vectorisedcandidatelines = np.asarray(
                [
                    [
                        [0, cgrad[1]],
                        np.asarray([1, cgrad[0]]) / (1 + cgrad[0]**2) **.5
                    ]
                    for cgrad in candidategrads
                ]
            )

            

        highestvectorcandidate = [max(
            vectorisedcandidatelines,
            key=lambda v: v[0][1] + v[0][1] + target.shape[1] * v[1][1]
        )]


        v = highestvectorcandidate[0]
        highestline = [(
            v[0][0],
            v[0][1],
            v[0][0] + target.shape[1] * v[1][0],
            v[0][1] + target.shape[1] * v[1][1],
            
        )]

        derivedlines = np.asarray(highestline, dtype = np.float32)


        translatedvectors = [
            [
                np.matmul(np.asarray([[np.cos(thet), -np.sin(thet)], [np.sin(thet), np.cos(thet)]]), np.asarray(vline[0])) + np.asarray(translation),
                np.matmul(np.asarray([[np.cos(thet), -np.sin(thet)], [np.sin(thet), np.cos(thet)]]), vline[1]) 
            ]
            for vline in highestvectorcandidate
        ]

        print(highestvectorcandidate)
        print("ye", np.sum(highestvectorcandidate[0][1] ** 2) ** 0.5, )

        print("tr", np.sum(translatedvectors[0][1] ** 2) ** 0.5, )

        cumvectors += translatedvectors

        cumderivedlines += [
            [(
                v[0][0],
                v[0][1],
                v[0][0] + 10000 * v[1][0],
                v[0][1] + 10000 * v[1][1],
                
            )]
            for v in translatedvectors
        ]
    
    return cumvectors

def vectorIntersect(vecA, vecB):
    if vecA[1][0] * vecB[1][1] - vecA[1][1] * vecB[1][0] == 0:
        return None
    j = (vecA[1][0] * vecA[0][1] - vecA[1][1] * vecA[0][0] + vecA[1][1] * vecB[0][0] - vecA[1][0] * vecB[0][1]) / (vecA[1][0] * vecB[1][1] - vecA[1][1] * vecB[1][0])
    return np.asarray([vecB[0][0] + j * vecB[1][0], vecB[0][1] + j * vecB[1][1]])

def optimumVertices(edges):
    vertices = [None, None, None, None]
    finvertices = [None, None, None, None]

    bestarea = 0

    for l in range(4):
        this = edges[l]
        this_perp = [this[0], np.array([this[1][1], -this[1][0]])]
        adjcw = edges[l - 3]
        adjccw = edges[l - 1]
        opp = edges[l - 2]
        opp_perp = [opp[0], np.array([opp[1][1], -opp[1][0]])]

        cw = np.dot(edges[l][1], adjcw[1])
        ccw = np.dot(edges[l][1], adjccw[1])

        print(l, cw, ccw)


        if np.dot(this[1], opp_perp[1]) > 0:
            if ccw > 0:
                continue
            vertices[1] = vectorIntersect([vectorIntersect(adjcw, this), this_perp[1]], opp)
            vertices[2] = vectorIntersect(adjccw, [vertices[1], this[1]])
        else:
            if cw > 0:
                continue
            vertices[2] = vectorIntersect([vectorIntersect(adjccw, this), this_perp[1]], opp)
            vertices[1] = vectorIntersect(adjcw, [vertices[2], this[1]])

        vertices[0] = vectorIntersect(this, [vertices[1], this_perp[1]])

        vertices[3] = vectorIntersect(this, [vertices[2], this_perp[1]])

        vertices = np.roll(np.array(vertices), l + 1, 0)

        

        area = np.product(vertices[2] - vertices[0])

        print(vertices, area, bestarea)
        if area > bestarea:
            finvertices = vertices
            ourbox = np.asarray(list(map(lambda x: np.concatenate(x), [vertices[0:2], vertices[1:3], vertices[2:4], [vertices[3], vertices[0]]])))

    return finvertices

def cropToThreeVert(vertices, tocrop):
    src = np.asarray(vertices, dtype=np.float32)

    dst = np.asarray([[0, 0], [tocrop.shape[1], 0], [tocrop.shape[1], tocrop.shape[0]], [0, tocrop.shape[0]]], dtype=np.float32)

    affine = cv2.getAffineTransform(src[:3], dst[:3])

    desired = cv2.warpAffine(
        tocrop,
        affine,
        [int(np.sum((vertices[0] - vertices[1]) ** 2) ** 0.5), int(np.sum((vertices[1] - vertices[2]) ** 2) ** 0.5)]
    )

    return desired[10: -10, 10: -10]

def processAndSaveImage(raw, ifrraw, outdir="testing", i=0):
    flipped = normaliseAndInvert(raw)

    typedmask = fixDustMask(raw, ifrraw)

    typedreal = (flipped * (2 ** 8 - 1)).astype(np.uint8)
    dedusted = cv2.inpaint(typedreal, typedmask, 3, cv2.INPAINT_TELEA)

    edges = findEdges(np.average(raw, 2))

    vertices = optimumVertices(edges)

    croppedflipped = cropToThreeVert(vertices, dedusted)

    try:
        os.mkdir(outdir)
    except:
        pass

    cv2.imwrite(f"{outdir}/outputfile"+str(i)+".jpg", cv2.cvtColor(croppedflipped, cv2.COLOR_RGB2BGR))

def loadfileAndProcess(folderpath, filename, outdir="testing", i=0):
    # for i in sorted(glob.glob("../photos/positives/France 1976/*.NEF")):
    raw = cv2.imread(f"{folderpath}/normal/{filename}")

    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    raw = raw/raw.max()

    ifrraw = cv2.imread(f"{folderpath}/infra/{filename}")

    ifrraw = cv2.cvtColor(ifrraw, cv2.COLOR_BGR2RGB)
    ifrraw = ifrraw/ifrraw.max()


    ifrraw = np.add.reduce(ifrraw, axis = 2 ) / 3
    
    processAndSaveImage(raw, ifrraw, outdir, i)

    return "hi"


if __name__ == "__main__":
    fkey = "negs 1978"

    folderpath = f"../photos/{fkey}"
    outdir = f"../photos/procd/{fkey}"
    fnames = listfiles(folderpath)

    number = len(fnames)
    outputs = []


    gridsize = int(number**0.5//1+(number**0.5%1>0)*1)

    with Pool(processes = 7) as pool:
        results = [
            pool.apply_async(
                loadfileAndProcess,
                (folderpath, fnames[i], outdir, i)
            ) for i in range(len(fnames))[:number]
        ]

        [result.wait() for result in results]
    # loadfileAndProcess(folderpath, fnames[0], 0)






