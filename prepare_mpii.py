"""Extract all human joint annotations from MPII and rectangular bounding
boxes of the corresponding image.
Patrick.Esser@gmx.net"""

import os
from tqdm import tqdm
import logging
import argparse
import sys
import urllib.request
import pickle
from zipfile import ZipFile
import tarfile
import numpy as np
import scipy.io
import cv2


files = {
        "mpii_human_pose_v1.tar.gz": "http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz",
        "mpii_human_pose_v1_u12_2.zip": "http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip"}


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
            #targets = f.getnames()
            targets = [os.path.join(data_dir, "images")] # full check too slow
            if not all([os.path.exists(target) for target in targets]):
                print("Extracting {}".format(path))
                f.extractall(data_dir)
            else:
                print("Skipping extraction of {}".format(path))
    elif path.endswith(".zip"):
        with ZipFile(path) as f:
            #targets = [f.getnames()[0]]
            targets = [os.path.join(data_dir, "mpii_human_pose_v1_u12_2")] # full check too slow
            if not all([os.path.exists(target) for target in targets]):
                print("Extracting {}".format(path))
                f.extractall(data_dir)
            else:
                print("Skipping extraction of {}".format(path))
    return targets


def extract_annotations(annotations):
    # only training images contain joint annotations
    # can extract 17408 joint annotations from 18079 training images
    # seperated persons would be nice but there are only 2161

    # img_train
    py_img_train = annotations["RELEASE"][0,0]["img_train"].squeeze().astype(bool)

    # annolist
    annolist = annotations["RELEASE"][0,0]["annolist"].squeeze()
    py_annodict = dict()
    for i in tqdm(range(annolist.shape[0])):
        if py_img_train[i]:
            img_fname = annolist[i]["image"][0,0][0][0]
            annorect = annolist[i]["annorect"]
            vididx = annolist[i]["vididx"][0,0]
            py_annorect = dict()
            for a in range(annorect.shape[1]):
                try:
                    # rough scale and position
                    scale = annorect[0,a]["scale"][0,0]
                    objpos_x = annorect[0,a]["objpos"][0,0][0][0,0]
                    objpos_y = annorect[0,a]["objpos"][0,0][1][0,0]

                    # joint annotations
                    point = annorect[0,a]["annopoints"]["point"][0,0].squeeze()
                    py_point = []
                    for p in point:
                        x = p["x"][0,0]
                        y = p["y"][0,0]
                        id_ = p["id"][0,0]
                        py_point.append({
                            "x": x,
                            "y": y,
                            "id": id_})

                    py_annorect[a] = {
                        "scale": scale,
                        "objpos": (objpos_x, objpos_y),
                        "annopoints": py_point}
                except:
                    pass

            if len(py_annorect) > 0:
                py_annodict[i] = {
                    "image": img_fname,
                    "annorect": py_annorect,
                    "vididx": vididx}

    return py_annodict


def mark_split(labels):
    vidindices = list()
    for k, v in labels.items():
        vidindices.append(v["vididx"])
    unique_vi = np.unique(vidindices)
    np.random.seed(42)
    unique_vi = np.random.permutation(unique_vi)
    n_vids = unique_vi.shape[0]
    split_index = int(0.9 * n_vids)
    train_vi = unique_vi[:split_index]
    valid_vi = unique_vi[split_index:]
    for k in labels:
        train = labels[k]["vididx"] in train_vi
        labels[k]["train"] = train


def load_labels(folder):
    labels_file = os.path.join(folder, "labels.p")
    if not os.path.exists(labels_file):
        print("Loading labels from matlab file.")
        fname = os.path.join(folder, "mpii_human_pose_v1_u12_1.mat")
        labels = scipy.io.loadmat(
                fname,
                mat_dtype = True,
                squeeze_me = False,
                struct_as_record = True)
        labels = extract_annotations(labels)
        mark_split(labels)
        with open(labels_file, "wb") as f:
            pickle.dump(labels, f)
    else:
        print("Loading labels from pickle file.")

    with open(labels_file, "rb") as f:
        labels = pickle.load(f)

    return labels


jo = ["rankle", "rknee", "rhip", "lhip", "lknee", "lankle", "rwrist", "relbow", "rshoulder", "lshoulder", "lelbow", "lwrist", "chead"]
idt = {
        0: jo.index("rankle"),
        1: jo.index("rknee"),
        2: jo.index("rhip"),
        3: jo.index("lhip"),
        4: jo.index("lknee"),
        5: jo.index("lankle"),
        9: jo.index("chead"),
        10: jo.index("rwrist"),
        11: jo.index("relbow"),
        12: jo.index("rshoulder"),
        13: jo.index("lshoulder"),
        14: jo.index("lelbow"),
        15: jo.index("lwrist")}
n_joints = len(jo)

def prepare_joints(orig_joints):
    points = orig_joints["annopoints"]
    prep_joints = np.zeros((n_joints, 2))
    for p in points:
        jid = p["id"]
        if jid in idt:
            prep_joints[idt[jid], 0] = p["x"]
            prep_joints[idt[jid], 1] = p["y"]
    return prep_joints


def prepare_img(img, joints):
    # unlabeled / invisible joints are set to zero in mpi
    # here only annotations with a margin of m are considered as valid but
    # even with this some invalid annotations slip through...
    wh = np.array([img.shape[1], img.shape[0]])
    m = 5
    valid_j = (0 + m < joints) & (joints < wh - 1 - m)
    valid_j = np.all(valid_j, axis = 1, keepdims = True)
    valid_j = np.repeat(valid_j, 2, axis = 1)
    joints[~valid_j] = -1000.0
    if not valid_j.all():
        return None, None, None

    # bounding box
    minx = int(np.min(joints[:,0][valid_j[:,0]]))
    maxx = int(np.max(joints[:,0][valid_j[:,0]]))
    miny = int(np.min(joints[:,1][valid_j[:,1]]))
    maxy = int(np.max(joints[:,1][valid_j[:,1]]))

    # make sure even width
    bbw = maxx - minx
    if not bbw % 2 == 0:
        maxx = maxx + 1
    bbw = maxx - minx
    # make sure even height
    bbh = maxy - miny
    if not bbh % 2 == 0:
        maxy = maxy + 1
        bbh = maxy - miny

    # expand smaller dimension to have rectangular bbox
    if bbw < bbh:
        p = (bbh - bbw) // 2
        maxx = maxx + p
        minx = minx - p
        bbw = maxx - minx
    elif bbh < bbw:
        p = (bbw - bbh) // 2
        maxy = maxy + p
        miny = miny - p
        bbh = maxy - miny

    # add padding
    padding = int(0.1 * bbw)
    minx = minx - padding
    maxx = maxx + padding
    miny = miny - padding
    maxy = maxy + padding

    # visualize bbox
    #cv2.rectangle(img, (minx, miny), (maxx, maxy), thickness = 5, color = (255,0,0))

    # transformation to bounding box
    src = np.float32([
        [minx, miny], [maxx, miny],
        [minx, maxy], [maxx, maxy]])
    dst = np.float32([
        [0, 0], [255, 0],
        [0,255], [255,255]])
    M = cv2.getPerspectiveTransform(src, dst)

    # transform image
    img = cv2.warpPerspective(img, M, (256, 256), flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_REPLICATE)

    # transform joints
    joint_dst = cv2.perspectiveTransform(np.expand_dims(joints, 0), M)
    joints = np.squeeze(joint_dst)
    # mark again nonvalid joints with negative values
    # since they could now have valid coordinates
    joints[~valid_j] = -1000.0

    # generate mask as poor man's segmentation
    masks = 3*[None]
    for i in range(3):
        masks[i] = np.zeros(img.shape[:2], dtype = "uint8")

    body = ["lhip", "lshoulder", "rshoulder", "rhip"]
    body_pts = np.array([[joints[jo.index(part),:] for part in body]], dtype = np.int32)
    if np.min(body_pts) >= 0:
        cv2.fillPoly(masks[1], body_pts, 255)
    else:
        return None, None, None

    head = ["lshoulder", "chead", "rshoulder"]
    head_pts = np.array([[joints[jo.index(part),:] for part in head]], dtype = np.int32)
    if np.min(head_pts) >= 0:
        cv2.fillPoly(masks[2], head_pts, 255)
    else:
        return None, None, None

    lines = [[
        ("rankle", "rknee"),
        ("rknee", "rhip"),
        ("rhip", "lhip"),
        ("lhip", "lknee"),
        ("lknee", "lankle") ], [
            ("rhip", "rshoulder"),
            ("rshoulder", "relbow"),
            ("relbow", "rwrist"),
            ("rhip", "lhip"),
            ("rshoulder", "lshoulder"),
            ("lhip", "lshoulder"),
            ("lshoulder", "lelbow"),
            ("lelbow", "lwrist")], [
                ("rshoulder", "chead"),
                ("rshoulder", "lshoulder"),
                ("lshoulder", "chead")]]
    for i in range(len(lines)):
        for j in range(len(lines[i])):
            line = [jo.index(lines[i][j][0]), jo.index(lines[i][j][1])]
            if np.min(joints[line]) >= 0:
                a = tuple(np.int_(joints[line[0]]))
                b = tuple(np.int_(joints[line[1]]))
                cv2.line(masks[i], a, b, color = 255, thickness = 50)

    for i in range(3):
        masks[i] = cv2.GaussianBlur(masks[i], (51,51), 0)
        maxmask = np.max(masks[i])
        if maxmask > 0:
            masks[i] = masks[i] / maxmask
    mask = np.stack(masks, axis = -1)
    mask = np.uint8(255 * mask)

    # apply mask
    apply_mask = False
    if apply_mask:
        mask = mask / 255
        # combine mask
        mask = np.max(mask, axis = 2)[:,:,None]
        img = img * mask

    return img, mask, joints


def prepare(target_path, img_path, labels):
    os.makedirs(target_path, exist_ok = True)
    train = list()
    imgs = list()
    masks = list()
    joints = list()
    joint_order = jo
    for i, label in tqdm(enumerate(labels.values())):
        img_fname = label["image"]
        base_fname = img_fname.rsplit(".", maxsplit = 1)[0]
        orig_img_fname = os.path.join(img_path, img_fname)
        orig_joint_list = label["annorect"]
        for a, orig_joints in enumerate(orig_joint_list.values()):
            prep_img_fname = base_fname + "_{:02}.png".format(a)
            prep_mask_fname = base_fname + "_{:02}_mask.png".format(a)

            prep_img_path = os.path.join(target_path, prep_img_fname)
            prep_mask_path = os.path.join(target_path, prep_mask_fname)

            prep_joints = prepare_joints(orig_joints)
            orig_img = cv2.imread(orig_img_fname)
            prep_img, prep_mask, prep_joints = prepare_img(orig_img, prep_joints)

            if prep_img is not None:
                cv2.imwrite(prep_img_path, prep_img)
                cv2.imwrite(prep_mask_path, prep_mask)

                imgs.append(prep_img_fname)
                masks.append(prep_mask_fname)
                joints.append(prep_joints)
                train.append(label["train"])

    index = {"imgs": imgs, "masks": masks, "joints": joints, "train": train, "joint_order": joint_order}
    index_fname = os.path.join(target_path, "index.p")
    with open(index_fname, "wb") as f:
        pickle.dump(index, f)

    print("Prepared {} annotations.".format(len(train)))
    print("Training: {}".format(train.count(True)))
    print("Validation: {}".format(train.count(False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required = True, help = "path to store data at")
    parser.add_argument("--out_dir", required = True, help = "path to store prepared data at")
    opt = parser.parse_args()

    os.makedirs(opt.data_dir, exist_ok = True)
    files = download_files(files, opt.data_dir)
    for f in files:
        extract_data(f, opt.data_dir)

    labels_folder = os.path.join(opt.data_dir, "mpii_human_pose_v1_u12_2")
    assert(os.path.exists(labels_folder))
    labels = load_labels(labels_folder)
    print("Loaded {} labels.".format(len(labels)))
    img_dir = os.path.join(opt.data_dir, "images")
    prepare(opt.out_dir, img_dir, labels)
