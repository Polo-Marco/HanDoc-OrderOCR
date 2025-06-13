import cv2
import numpy as np
import json
from PIL import ImageFont, ImageDraw, Image 
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
font = ImageFont.truetype("./font/NotoSerifCJKtc-ExtraLight.otf", 30)
#prepocess of rec
def crop_img(img,poly_pts,pad_color="white",rotate=None):
    rotate_map = {90:cv2.ROTATE_90_CLOCKWISE,180:cv2.ROTATE_180,270:cv2.ROTATE_90_COUNTERCLOCKWISE}#mapping to cv2 rotate code
    pts = np.array(poly_pts,dtype=np.int32)
    ## (1) Crop the bounding rect
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = img[y:y+h, x:x+w].copy()

    ## (2) make mask
    pts = pts - pts.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    ## (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)
    ## (4) add the white background
    bg = np.ones_like(croped, np.uint8)*255
    cv2.bitwise_not(bg,bg, mask=mask)
    dst2 = bg+ dst
    if pad_color == "white":
        if rotate:
            return cv2.rotate(dst2,rotate_map[rotate])
        return dst2
    if rotate:
        return cv2.rotate(dst,rotate_map[rotate])
    return dst

#visualize det & rec
def read_det(det_path):
    with open(det_path,"r") as f:
        return json.loads(f.readline().split("\t")[1])
def read_rec(rec_path):
    rec_anno={}
    with open(rec_path,"r") as f:
        line = f.readline()
        while line:
            line = line.split("\t")
            file,anno = line[0][:-4],line[1]
            file = file.split("_")[-1]
            rec_anno[file]=anno
            line = f.readline()
    return rec_anno
def read_order(filename,order_path):
    with open(order_path+"prediction.json","r") as f:
        data = json.load(f)[filename[:-4]]
    return {data[key]:key for key in data}#reverse annotation
        
def return_tl_rb(boxes):
    xs = [box[0] for box in boxes]
    ys = [box[1] for box in boxes]
    return (min(xs),min(ys)),(max(xs),max(ys))

def vis_det_rec(filename,ori_img_path,det_path,rec_path,order_path):
    rec_anno = read_rec(rec_path)
    det_anno = read_det(det_path)
    order_anno = read_order(filename,order_path)
    seq_text=list(range(len(order_anno.keys())))
    image = Image.open(ori_img_path).convert('RGB')
    draw = ImageDraw.Draw(image, 'RGB')
    for idx,annotation in enumerate(det_anno):
        poly = [elem for elem1 in annotation["points"] for elem in elem1 ]#flat 2d cordinates to 1d
        tl,br = return_tl_rb(annotation["points"])
        poly_outline=(255,0,0)
        text = "\n".join([str(order_anno[idx])]+list(rec_anno[str(idx)]))
        seq_text[order_anno[idx]]="".join(rec_anno[str(idx)])
        #poly_outline=(255,0,0)
        poly_center = (tl[0]+(abs(tl[0]-br[0]))/2,tl[1]+abs((tl[1]-br[1]))/2)
        poly_center = (poly_center[0]+(abs(tl[0]-br[0])/2),poly_center[1])
        #print(poly_center)
        # Draw sentence boxes
        draw.polygon(poly,
                     outline=poly_outline,
                     width=3)
        l_u_point=poly_center
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((0, 0), text, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
        else:
            w, h = draw.textsize(text, font=font)
        draw.rectangle(
            (l_u_point[0] - w/2,
             l_u_point[1] - h/2,
             l_u_point[0] + w/2,
             l_u_point[1] + h/2),
            fill=(64, 64, 64, 0),
        )
        draw.text(
            (l_u_point[0] - w/2,
            l_u_point[1] - h/2),
            text=text,
            fill=(255, 255, 255, 255),
            font=font
        )
    return image,seq_text
#eading order detection
def order_preproc(img_path,filenames,det_path,save_path):
    det_anno={}
    for file in filenames:
        height,width  = cv2.imread(img_path+file).shape[:2]
        det_anno[file[:-4]]= {"line_label":[i["points"] for i in read_det(det_path)],
                        "width":width,"height":height}
    with open(save_path+"predict.json",'w') as f:
            json.dump(save_pair(pair_data(det_anno)),f)
def pair_data(data_dict):
    '''
    input: imgs dict
    output: pair-wised dataset with sentence and text as key
    '''
    char_pair,sent_pair = {},{}
    for img in tqdm(data_dict.keys()):
        w,h = data_dict[img]["width"],data_dict[img]["height"]
        features,pairs = pair(data_dict[img]["line_label"],w,h)
        sent_pair[img]= {"features": features,
                         "index": pairs,
                         "width": w,
                         "height": h
                         }
    return sent_pair
def flatten_lst(seq):
    return [seq_1d for seq_2d in seq for seq_1d in seq_2d ]
def pair(seq,width,height):
    re_lst = []
    pair = []
    if width ==None:
        width =1
    if height == None:
        height=1
    for i in range(len(seq)):
        for j in range(len(seq)):
            if i==j:
                continue
            feature = flatten_lst(seq[i])+flatten_lst(seq[j])
            normalized_feature = [elem if idx%2 else elem for idx,elem in enumerate(feature) ]
            re_lst.append(normalized_feature)
            pair.append([i,j])
    return re_lst,pair
def save_pair(data):
    re_lst=[]
    for img in data:
        for feature,idx in zip(data[img]["features"],data[img]["index"]):
            re_lst.append({"feature":feature,
                          "label":0,
                          "index":idx,
                          "image":img,
                          "width": data[img]["width"],
                         "height": data[img]["height"]})
    return re_lst
if __name__ == "__main__":
    read_order("./examples/YB_24_204.jpg","order_det/")
    #vis_det_rec("./examples/YB_24_204.jpg","./predicts_db.txt","predicts_svtr_tiny_ch_all.txt")
    #order_preproc("./processed/",["32-V023P0662.jpg"],"predicts_db.txt","./order_det/")
    print("end utils")
