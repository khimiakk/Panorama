import cv2
import numpy as np
import matplotlib.pyplot as plt

def detectFeaturesAndMatch(img1, img2, nFeaturesReturn=30):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return np.array([]), None, None

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    correspondences = []
    for match in matches[:nFeaturesReturn]:
        correspondences.append((kp1[match.queryIdx].pt, kp2[match.trainIdx].pt))

    src = np.float32([m[0] for m in correspondences]).reshape(-1, 1, 2)
    dst = np.float32([m[1] for m in correspondences]).reshape(-1, 1, 2)

    return np.array(correspondences), src, dst

#TESTING whether the function works correctly

def show_orb_keypoints(img, title="ORB keypoints"):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=500)

    kp, des = orb.detectAndCompute(gray, None)

    img_kp = cv2.drawKeypoints(
        img, kp, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    print(f"{title}:")
    print("Number of keypoints:", len(kp))
    print("Descriptors shape:", None if des is None else des.shape)

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

    return kp, des
# box = cv2.imread("box.png")
# kp1, des1 = show_orb_keypoints(box, "ORB keypoints on box.png")

# scene = cv2.imread("box-in-scene.png")
# kp2, des2 = show_orb_keypoints(scene, "ORB keypoints on box-in-scene.png")

def detectFeaturesAndMatch(img1, img2, nFeaturesReturn=30):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return None, None, None

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    return kp1, kp2, matches[:nFeaturesReturn]


def show_matches(img1, img2, kp1, kp2, matches, title="ORB matches"):
    matched_img = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


# box = cv2.imread("box.png")
# scene = cv2.imread("box-in-scene.png")
# if box is None:
#     print("box.png not loaded")
# elif scene is None:
#     print("box-in-scene.png not loaded")
# else:
#     kp1, kp2, matches = detectFeaturesAndMatch(box, scene, nFeaturesReturn=40)

#     show_matches(box, scene, kp1, kp2, matches, title="Top ORB matches: box -> scene")