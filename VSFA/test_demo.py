"""Test Demo for Quality Assessment of In-the-Wild Videos, ACM MM 2019"""
#
# Author: Dingquan Li
# Email: dingquanli AT pku DOT edu DOT cn
# Date: 2018/3/27
#
import torch
from torchvision import transforms
import skvideo.io
from PIL import Image
import numpy as np
from VSFA import VSFA
from CNNfeatures import get_features
from argparse import ArgumentParser
import time


if __name__ == "__main__":
    parser = ArgumentParser(description='"Test Demo of VSFA')
    parser.add_argument('--model_path', default='models/VSFA.pt', type=str,
                        help='model path (default: models/VSFA.pt)')
    parser.add_argument('--video_path', default='./test.mp4', type=str,
                        help='video path (default: ./test.mp4)')
    parser.add_argument('--video_format', default='RGB', type=str,
                        help='video format: RGB or YUV420 (default: RGB)')
    parser.add_argument('--video_width', type=int, default=None,
                        help='video width')
    parser.add_argument('--video_height', type=int, default=None,
                        help='video height')

    parser.add_argument('--frame_batch_size', type=int, default=32,
                        help='frame batch size for feature extraction (default: 32)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start = time.time()

    # data preparation
    assert args.video_format == 'YUV420' or args.video_format == 'RGB'
    if args.video_format == 'YUV420':
        video_data = skvideo.io.vread(args.video_path, args.video_height, args.video_width, inputdict={'-pix_fmt': 'yuvj420p'})
    else:
        video_data = skvideo.io.vread(args.video_path)

    video_length = video_data.shape[0]
    video_channel = video_data.shape[3]
    video_height = video_data.shape[1]
    video_width = video_data.shape[2]
    transformed_video = torch.zeros([video_length, video_channel, video_height, video_width])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for frame_idx in range(video_length):
        frame = video_data[frame_idx]
        frame = Image.fromarray(frame)
        frame = transform(frame)
        transformed_video[frame_idx] = frame

    print('Video length: {}'.format(transformed_video.shape[0]))

    # feature extraction
    features = get_features(transformed_video, frame_batch_size=args.frame_batch_size, device=device)
    features = torch.unsqueeze(features, 0)  # batch size 1

    # quality prediction using VSFA
    model = VSFA()
    model.load_state_dict(torch.load(args.model_path))  #
    model.to(device)
    model.eval()
    with torch.no_grad():
        input_length = features.shape[1] * torch.ones(1, 1)
        outputs = model(features, input_length)
        y_pred = outputs[0][0].to('cpu').numpy()
        print("Predicted quality: {}".format(y_pred))

    end = time.time()

    print('Time: {} s'.format(end-start))
