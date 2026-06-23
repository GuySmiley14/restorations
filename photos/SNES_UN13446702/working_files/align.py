import cv2
import numpy as np

SOURCE = "top_shell_outside-before.jpg"
REFERENCE = "top_shell_outside-after.jpg"

source = cv2.imread(SOURCE)
reference = cv2.imread(REFERENCE)

g1 = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
g2 = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(10000)

kp1, des1 = orb.detectAndCompute(g1, None)
kp2, des2 = orb.detectAndCompute(g2, None)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

matches = matcher.knnMatch(des1, des2, k=2)

good = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

src_pts = np.float32(
    [kp1[m.queryIdx].pt for m in good]
).reshape(-1, 1, 2)

dst_pts = np.float32(
    [kp2[m.trainIdx].pt for m in good]
).reshape(-1, 1, 2)

H, mask = cv2.findHomography(
    src_pts,
    dst_pts,
    cv2.RANSAC,
    5.0
)

aligned = cv2.warpPerspective(
    source,
    H,
    (reference.shape[1], reference.shape[0])
)

cv2.imwrite("top_shell_outside-before(aligned).jpg", aligned)