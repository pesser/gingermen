"""Extract all human joint annotations from MSCOCO and rectangular bounding
boxes of the corresponding image.
Patrick.Esser@gmx.net"""

import sys
import os
from tqdm import tqdm
import logging
import argparse
import sys
import urllib.request
import pickle
from zipfile import ZipFile
import numpy as np
import scipy.io
import skimage.io
import cv2
import json


files = {
        "person_keypoints_trainval2014.zip": "http://msvocds.blob.core.windows.net/annotations-1-0-3/person_keypoints_trainval2014.zip"}


def dl_progress(count, block_size, total_size):
    """Progress bar used during download."""
    if total_size == -1:
        if count == 0:
            sys.stdout.write("Unknown size of download.\n")
    else:
        length = 50
        current_size = count * block_size
        done = current_size * length // total_size
        togo = length - done
        prog = "[" + done * "=" + togo * "-" + "]"
        sys.stdout.write(prog)
        if(current_size < total_size):
            sys.stdout.write("\r")
        else:
            sys.stdout.write("\n")
    sys.stdout.flush()


def download_files(files, target_dir):
    fnames = list()
    for fname, url in files.items():
        fname = os.path.join(target_dir, fname)
        fnames.append(fname)
        if not os.path.isfile(fname):
            print("Downloading {}".format(fname))
            urllib.request.urlretrieve(url, fname, reporthook = dl_progress)
        else:
            print("Found {}. Skipping download.".format(fname))
    return fnames


def extract_data(path, data_dir):
    """Extract zip file if not already extracted."""
    if path.endswith(".tar.gz"):
        with tarfile.open(path) as f:
            targets = f.getnames()
            if not all([os.path.exists(target) for target in targets]):
                print("Extracting {}".format(path))
                f.extractall(data_dir)
            else:
                print("Skipping extraction of {}".format(path))
    elif path.endswith(".zip"):
        with ZipFile(path) as f:
            targets = [os.path.join(data_dir, x) for x in f.namelist()]
            if not all([os.path.exists(target) for target in targets]):
                print("Extracting {}".format(path))
                f.extractall(data_dir)
            else:
                print("Skipping extraction of {}".format(path))
    return targets


def get_img_by_id(id_, anno):
    for img in anno["images"]:
        if img["id"] == id_:
            return img


def prepare(kps_path, out_path, is_train):
    print("Loading keypoints.")
    coco = COCO(kps_path)
    with open(kps_path, "r") as f:
        kps = json.load(f)

    n_kps = len(kps["annotations"])
    print("Loaded {} annotations.".format(n_kps))

    person_cat = [c for c in kps["categories"] if c["name"] == "person"][0]
    person_cat_id = person_cat["id"]
    kp_names = person_cat["keypoints"]
    print("Person keypoints: {}".format(kp_names))
    print("Filtering keypoints by category person (id = {}).".format(person_cat_id))
    person_kps = [k for k in kps["annotations"] if k["category_id"] == person_cat_id]
    n_person_kps = len(person_kps)
    print("Found {} ({} / {} = {:.4}%) person annotations.".format(n_person_kps, n_person_kps, n_kps, 100*n_person_kps/n_kps))

    # without head kps: 28344
    # with nose       : 22965
    # with eyes       : 17230
    # with eyes+nose  : 17104
    required_kp_names = kp_names[5:] + ["nose"]# + ["left_eye", "right_eye"]
    required_kp_indices = list(kp_names.index(name) for name in required_kp_names)
    print("Filtering keypoints by visibility labeled for: {}".format(required_kp_names))
    visible_kps = list()
    for i in range(n_person_kps):
        k = person_kps[i]
        # 17 keypoints, xyv
        keypoints = np.array(k["keypoints"]).reshape(17, 3)
        all_labeled = (keypoints[:,2] != 0)[required_kp_indices].all()
        if all_labeled:
            visible_kps.append(k)
    n_visible_kps = len(visible_kps)
    print("Found {} ({} / {} = {:.4}%) fully labeled person annotations.".format(n_visible_kps, n_visible_kps, n_kps, 100*n_visible_kps/n_kps))

    # translate joint names and fix order
    jo = ["rankle", "rknee", "rhip", "lhip", "lknee", "lankle", "rwrist", "relbow", "rshoulder", "lshoulder", "lelbow", "lwrist", "cnose"]
    jn = kp_names
    jt = [
            jn.index("right_ankle"), jn.index("right_knee"),
            jn.index("right_hip"), jn.index("left_hip"),
            jn.index("left_knee"), jn.index("left_ankle"),
            jn.index("right_wrist"), jn.index("right_elbow"),
            jn.index("right_shoulder"), jn.index("left_shoulder"),
            jn.index("left_elbow"), jn.index("left_wrist"),
            jn.index("nose")]

    # build index
    imgs = list()
    masks = list()
    joints_list = list()
    train = list()
    joint_order = jo

    for i in tqdm(range(n_visible_kps)):
        k = visible_kps[i]
        assert(not k["iscrowd"])
        assert(k["category_id"] == person_cat_id)

        # keypoints
        keypoints = np.array(k["keypoints"]).reshape(17,3)
        joints = np.float32(keypoints[:,:2])

        # image
        image_id = k["image_id"]
        img = get_img_by_id(image_id, kps)
        try:
            I = skimage.io.imread(img["coco_url"])
        except:
            print("Could not load {}".format(img["coco_url"]))
            continue

        # mask
        mask = coco.annToMask(k)

        # bbox
        x,y,w,h = k["bbox"]
        # make even dimensions
        if not w % 2 == 0:
            w = w + 1
        if not h % 2 == 0:
            h = h + 1
        # make rectangular bbox
        if w < h:
            p = (h - w) // 2
            x = x - p
            w = h
        elif h < w:
            p = (w - h) // 2
            y = y - p
            h = w
        # transform to bbox
        src = np.float32([
            [x, y], [x+w, y],
            [x, y+h], [x+w, y+h]])
        dst = np.float32([
            [0, 0], [255, 0],
            [0,255], [255,255]])
        M = cv2.getPerspectiveTransform(src, dst)

        transform = True
        if transform:
            I = cv2.warpPerspective(I, M, (256, 256), flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_REPLICATE)
            mask = cv2.warpPerspective(mask, M, (256, 256), flags = cv2.INTER_LINEAR)
            joints = np.expand_dims(joints, axis = 0)
            joints = cv2.perspectiveTransform(joints, M)
            joints = np.squeeze(joints)
        else:
            # draw the bounding box
            a = tuple(np.int_(src[0]))
            b = tuple(np.int_(src[3]))
            cv2.rectangle(I, a, b, (0,0,255))

        annotate = False
        if annotate:
            for jidx in required_kp_indices:
                cv2.circle(I, tuple(joints[jidx,:]), 3, (255,0,0))

        # save img
        prefix = "train_" if is_train else "valid_"
        out_fname = prefix + "{:06}.png".format(i)
        out_fname_path = os.path.join(out_path, out_fname)
        skimage.io.imsave(out_fname_path, I)

        # save mask
        mask = np.uint8(255 * mask)
        mask_fname = prefix + "{:06}_mask.png".format(i)
        mask_fname_path = os.path.join(out_path, mask_fname)
        skimage.io.imsave(mask_fname_path, mask)

        # sort joints
        joints = joints[jt,:]

        imgs.append(out_fname)
        masks.append(mask_fname)
        joints_list.append(joints)
        train.append(is_train)

    index = {"imgs": imgs, "masks": masks, "joints": joints_list, "train": train, "joint_order": joint_order}
    return index


def merge_indices(train_index, valid_index, out_path):
    assert(train_index["joint_order"] == valid_index["joint_order"])
    index = {
            "imgs": train_index["imgs"] + valid_index["imgs"],
            "masks": train_index["masks"] + valid_index["masks"],
            "joints": train_index["joints"] + valid_index["joints"],
            "train": train_index["train"] + valid_index["train"],
            "joint_order": train_index["joint_order"]}
    index_fname = os.path.join(out_path, "index.p")
    with open(index_fname, "wb") as f:
        pickle.dump(index, f)

    print("Prepared {} annotations.".format(len(index["train"])))
    print("Training: {}".format(index["train"].count(True)))
    print("Validation: {}".format(index["train"].count(False)))

    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required = True, help = "path to store data at")
    parser.add_argument("--out_dir", required = True, help = "path to store prepared data at")
    parser.add_argument("--coco_api", required = True, help = "path to coco python api")
    opt = parser.parse_args()

    sys.path.insert(0, opt.coco_api)
    from pycocotools.coco import COCO

    os.makedirs(opt.data_dir, exist_ok = True)
    os.makedirs(opt.out_dir, exist_ok = True)

    files = download_files(files, opt.data_dir)
    for f in files:
        extract_data(f, opt.data_dir)

    kps_path = os.path.join(opt.data_dir, "annotations", "person_keypoints_train2014.json")
    train_index = prepare(kps_path, opt.out_dir, is_train = True)
    kps_path = os.path.join(opt.data_dir, "annotations", "person_keypoints_val2014.json")
    valid_index = prepare(kps_path, opt.out_dir, is_train = False)

    index = merge_indices(train_index, valid_index, opt.out_dir)
