from mcap_db import McapDB
from pprint import pprint
import matplotlib.pyplot as plt
from tqdm import tqdm
from jsonargparse import ArgumentParser
import argparse
import cv2
import os


def mcap_to_images(folder, bag_name, image_topic, output_folder, prefix, split, interval, viz=True):
    file_path = f'{folder}/{bag_name}/{bag_name}_0.mcap'

    db = McapDB(file_path)
    pprint(db.topics())
    if '/tf' in db.topics():
        info = db.topic_info('/tf')
        messages = db.get(['/tf'], info.lifespan[0], info.lifespan[0] + 1)
        pprint(messages)

    info = db.topic_info(image_topic)
    messages = db.get([image_topic], info.lifespan[0], info.lifespan[1] + 1)
    count = len(messages)

    image = db.get_image(messages[count//2].data)

    if viz:
        plt.title(bag_name)
        plt.imshow(image)
        plt.show()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    count = 0
    for msg in tqdm(messages):
        if count % interval == 0:
            image = db.get_image(msg.data)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if split:
                h, w, _ = image.shape
                image = image[:, :w//2]
            #cv2.imwrite(f'{output_folder}/{bag_name}_{count}.jpg', image)
            cv2.imwrite(f'{output_folder}/{prefix}_frame{count:03d}.jpg', image)
        count += 1

if __name__ == '__main__':
    parser = ArgumentParser(default_config_files=['configs/jdb_player.yaml'])
    parser.add_argument('-f', '--folder', default=f'/home/jymeng/ros_ws/bags/', help='folder to bag')
    parser.add_argument('-b', '--bag', default='', help='bag')
    parser.add_argument('-s', '--split', action='store_true', default=False,
                        help='split stereo image and keep the left')
    parser.add_argument('-i', '--interval', type=int, default=1)
    parser.add_argument('-t', '--topic', default='/front/zed_node/left/image_rect_color', help='topic')
    parser.add_argument('-o', '--output', default='',  help='output')
    parser.add_argument('-p', '--prefix', default='',  help='prefix')
    parser.add_argument('-v', '--viz', action='store_true', default=False,
                        help='Visualize a dataset')
    args = parser.parse_args()

    mcap_to_images(args.folder, args.bag, args.topic, args.output, args.prefix, args.split, args.interval, args.viz)

