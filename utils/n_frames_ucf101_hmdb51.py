from __future__ import print_function, division
import os
import sys
import subprocess

def class_process(dir_path, class_name):
    class_path = os.path.join(dir_path, class_name)
    if not os.path.isdir(class_path):
        return

    for file_name in os.listdir(class_path):
        video_dir_path = os.path.join(class_path, file_name)
        image_indices = []
        for image_file_name in os.listdir(video_dir_path):
            #if 'image' not in image_file_name:
                #continue
            image_indices.append(int(image_file_name[:6]))

        if len(image_indices) == 0:
            print('no image files', video_dir_path)
            n_frames = 0
        else:
            image_indices.sort(reverse=True)
            n_frames = image_indices[0]
            print(video_dir_path, n_frames)

        n_frames_root = "/nfs_mount/code/ananth/code/Verisk/3D-ResNets-PyTorch/datasets/nframes"
        n_frames_path = os.path.join(n_frames_root, video_dir_path.split("/")[-2], video_dir_path.split("/")[-1])
        if not os.path.exists(n_frames_path):
            os.system("mkdir -p " + n_frames_path)
        with open(os.path.join(n_frames_path, 'n_frames'), 'w') as dst_file:
            dst_file.write(str(n_frames))


if __name__=="__main__":
    #dir_path = sys.argv[1]
    dir_path = "/nfs_mount/datasets/UCF101/UCF-101/frames_fps25_256"
    for j, class_name in enumerate(os.listdir(dir_path)):
        class_process(dir_path, class_name)
        print(j)
