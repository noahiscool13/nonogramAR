import cv2
import numpy as np
import dig_rec
import nonogram

img = cv2.imread('puzzles/img5.jpg')

# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
# TESSDATA_PREFIX = 'C:/Program Files (x86)/Tesseract-OCR'

def scale(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return img

ori = img.copy()
AR = img.copy()


img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (5, 5), 0)
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
img = cv2.bitwise_not(img, img)


def find_corners(img):
    ext_contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ext_contours = ext_contours[0] if len(ext_contours) == 2 else ext_contours[1]
    ext_contours = sorted(ext_contours, key=cv2.contourArea, reverse=True)
    for c in ext_contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        if len(approx) == 6:
            # cv2.drawContours(ori,approx,-1,(255,0,0),15)
            return approx

corners = find_corners(img)

corners = [x[0] for x in corners.tolist()][::-1]
print(corners)

right_down_ind = max(range(6),key=lambda x: sum(corners[x]))
print(right_down_ind)
corners = corners[(right_down_ind+4)%6:]+corners[:(right_down_ind+4)%6]
print(corners)

def intersect_lines(a1,a2,b1,b2):
    slope_a = (a2[1] - a1[1]) / (a2[0] - a1[0])
    off_a = a2[1] - slope_a * a2[0]
    slope_b = (b1[1] - b2[1]) / (b1[0] - b2[0])
    off_b = b1[1] - slope_b * b1[0]

    hidden_corner_x = (off_a - off_b) / (slope_b - slope_a)
    hidden_corner_y = slope_a * hidden_corner_x + off_a
    return (int(hidden_corner_x), int(hidden_corner_y))

hidden_corner = intersect_lines(corners[0],corners[1],corners[3],corners[4])


corner_dl = intersect_lines(corners[0],corners[5],corners[2],corners[3])
corner_tr = intersect_lines(corners[4],corners[5],corners[1],corners[2])

# TODO fix div by 0
# slope_a = (corners[1][1]-corners[0][1])/(corners[1][0]-corners[0][0])
# off_a = corners[1][1]-slope_a*corners[1][0]
# slope_b = (corners[3][1]-corners[4][1])/(corners[3][0]-corners[4][0])
# off_b = corners[3][1]-slope_b*corners[3][0]
#
#
# hidden_corner_x = (off_a-off_b)/(slope_b-slope_a)
# hidden_corner_y = slope_a*hidden_corner_x+off_a
# hidden_corner = (int(hidden_corner_x),int(hidden_corner_y))
print(hidden_corner)

growC = 7

# cv2.circle(ori,hidden_corner,20,(0,0,255),10)




width_A = np.sqrt(((corners[2][0] - corners[3][0]) ** 2) + ((corners[2][1] - corners[3][1]) ** 2))
width_B = np.sqrt(((corners[1][0] - hidden_corner[0]) ** 2) + ((corners[1][1] - hidden_corner[1]) ** 2))
width = max(int(width_A), int(width_B)) +2*growC

height_A = np.sqrt(((corners[1][0] - corners[2][0]) ** 2) + ((corners[1][1] - corners[2][1]) ** 2))
height_B = np.sqrt(((hidden_corner[0] - corners[3][0]) ** 2) + ((hidden_corner[1] - corners[3][1]) ** 2))
height = max(int(height_A), int(height_B))+2*growC

def grow(group,c):
    return [[group[0][0]-c,group[0][1]-c],[group[1][0]+c,group[1][1]-c],[group[2][0]+c,group[2][1]+c],[group[3][0]-c,group[3][1]+c]]

grid = cv2.getPerspectiveTransform(np.array(grow([hidden_corner,corners[1],corners[2],corners[3]],growC), dtype="float32"), np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]], dtype="float32"))
whole_board = cv2.warpPerspective(img, grid, (width, height))
imgb = cv2.warpPerspective(ori, grid, (width, height))
clean_cut = cv2.warpPerspective(ori, grid, (width, height))

imgb = cv2.cvtColor(imgb,cv2.COLOR_BGR2GRAY)
imgb = cv2.GaussianBlur(imgb, (3, 3), 0)
imgb = cv2.adaptiveThreshold(imgb, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 3)
imgb = cv2.bitwise_not(imgb, imgb)
imgb = cv2.dilate(imgb,np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8))




dl = 1

whole_board = cv2.dilate(whole_board,np.array([[0., dl, 0.], [dl, dl, dl], [0., dl, 0.]], np.uint8))
# whole_board = cv2.dilate(whole_board,np.array([[0., dl, 0.], [dl, dl, dl], [0., dl, 0.]], np.uint8))

boxes = []

# whole_board = cv2.resize(whole_board,(1143,1547))

contours,_ = cv2.findContours(whole_board, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("shp: ",imgb.shape)
i = 0
cnt = 0
digcnt = 0
for c in contours:
        area = cv2.contourArea(c)
        print(area)

        if 3000 > area > 800:



            # cv2.drawContours(clean_cut, contours, i, (0, 255, 0), 3)
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.070 * peri, True)
            if len(approx) == 4:

                # cv2.drawContours(clean_cut, contours, i, (0, 255, 0), 3)
                approx = approx.tolist()

                approx = sorted(approx,key=lambda x:x[0][1])

                tmp = []
                if approx[0][0][0]<approx[1][0][0]:
                    tmp.append(approx[0])
                    tmp.append(approx[1])
                else:
                    tmp.append(approx[1])
                    tmp.append(approx[0])

                if approx[2][0][0]<approx[3][0][0]:
                    tmp.append(approx[3])
                    tmp.append(approx[2])
                else:
                    tmp.append(approx[2])
                    tmp.append(approx[3])

                approx = tmp

                center = [sum(x[0][0] for x in approx)//4,sum(x[0][1] for x in approx)//4]
                num = 0

                # print(approx)

                grid = cv2.getPerspectiveTransform(
                    np.array(approx, dtype="float32"),
                    np.array([[0-3, 0-3], [50+3, 0-3], [50+3, 50+3], [0-3, 50+3]], dtype="float32"))
                block = cv2.warpPerspective(imgb, grid, (50, 50))
                if sum(cv2.mean(block)) > 3:
                    # dat = pytesseract.image_to_string(block, config='--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789')
                    # if dat!="\f":
                    #     print(dat[:-2])

                    ext_contours,_ = cv2.findContours(block, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    diglst = []

                    for cont in ext_contours:
                        x, y, w, h = cv2.boundingRect(cont)

                        if w*h<200 or w<=10 or h<=10:
                            continue

                        # print(w, h)
                        # block = cv2.rectangle(block, (x, y), (x + w, y + h), 255, 2)
                        digTrans = cv2.getPerspectiveTransform(
                            np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], dtype="float32"),
                            np.array([[0, 0], [50, 0], [50, 50], [0, 50]],
                                     dtype="float32"))
                        dig = cv2.warpPerspective(block, digTrans, (50, 50))
                        # print(dig.shape)

                        detected_dig = dig_rec.match(dig)

                        diglst.append([x,detected_dig])

                        # font = cv2.FONT_HERSHEY_SIMPLEX
                        # cv2.putText(clean_cut, str(detected_dig), (approx[0][0][0]+10,approx[0][0][1]+27), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
                        #
                        # # try:
                        # #     imgb[approx[0][0][1]:approx[0][0][1] + 50, approx[0][0][0]:approx[0][0][0] + 50] = dig
                        # # except Exception as e:
                        # #     print(approx[0][0],e)
                        #
                        # cv2.imshow("dig",dig)
                        # block = cv2.rectangle(block, (x, y), (x + w, y + h), 255, 2)
                        # cv2.imshow("blk",block)
                        # cv2.imwrite(f"test_digs/test{digcnt}.jpg",dig)
                        # digcnt+=1
                        # cv2.waitKey(0)
                    diglst.sort(key=lambda x:x[0])
                    found_nr = "".join([str(x[1]) for x in diglst])
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(clean_cut, found_nr, (approx[0][0][0]+10,approx[0][0][1]+27), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
                    try:
                        num = int(found_nr)
                    except:
                        cv2.imshow("err",block)

                    cnt += 1

                boxes.append([center,num])

            else:
                print(len(approx))



        i+=1

print(cnt)


# cv2.imshow('img',scale(clean_cut,70))
# cv2.waitKey(0)

boxes.sort(key=lambda x:x[0][1])

rows = []

y = boxes[0][0][1]
tmp = []

for b in boxes:
    if b[0][1]<y+5:
        tmp.append(b)
    else:
        rows.append(sorted(tmp,key=lambda x:x[0][0]))
        tmp = [b]
    y = b[0][1]
rows.append(sorted(tmp,key=lambda x:x[0][0]))



print("hi")

top_row = len(rows[0])
v_clue_len = 1
while len(rows[v_clue_len]) == top_row:
    v_clue_len+=1

print(top_row,v_clue_len)

left_col = len(rows)-v_clue_len
h_clue_len = len(rows[v_clue_len])-top_row

print(left_col,h_clue_len)

vclues = []
for x in range(top_row):
    tmp = []
    for y in range(v_clue_len):
        if rows[y][x][1]!=0:
            tmp.append(rows[y][x][1])
    vclues.append(tmp)

print(vclues)

hclues = []
for y in range(left_col):
    tmp = []
    for x in range(h_clue_len):
        if rows[y+v_clue_len][x][1]!=0:
            tmp.append(rows[y+v_clue_len][x][1])
    hclues.append(tmp)

print(hclues)

# print(boxes)

b = nonogram.Board(top_row,left_col,vclues,hclues)
b.solve()


res = cv2.cvtColor(np.array(b.board,dtype=np.uint8)*255,cv2.COLOR_GRAY2BGR)

pts1 = np.float32([[0,0],[res.shape[1]-1,0],[res.shape[1]-1,res.shape[0]-1],[0,res.shape[0]-1]])
pts2 = np.float32(grow([corners[5],corner_tr,corners[2],corner_dl],-30))


h, mask = cv2.findHomography(pts1, pts2)
res = cv2.warpPerspective(res, h, (ori.shape[1], ori.shape[0]),flags=cv2.INTER_NEAREST)

# cv2.imshow("res",np.array(b.board,dtype=np.uint8)*255)
# cv2.imshow("ww",scale(res,30))


# dat = pytesseract.image_to_data(img,config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
# print(dat)

# cv2.imshow('img1',scale(imgb,30))
# cv2.imshow('img2',scale(clean_cut,50))
# # b.show()
#
# cv2.circle(ori,corner_dl,20,(0,0,255),10)
# cv2.circle(ori,corner_tr,20,(0,0,255),10)
#
ori = cv2.bitwise_or(ori,res)
cv2.imshow('img2',scale(ori,30))
cv2.waitKey(0)