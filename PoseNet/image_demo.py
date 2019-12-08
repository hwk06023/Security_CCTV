# 필요한 모듈 및 파일을 임포트 한다.
import tensorflow as tf
import cv2
import time
import argparse
import os
import posenet
import sys

# 인자 값을 받을 수 있는 인스턴스 생성
parser = argparse.ArgumentParser()
# 입력받을 인자 값을 등록
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--notxt', action='store_true')
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--output_dir', type=str, default='./output')

# 입력받은 인자 값 출력
args = parser.parse_args()


def main():
    # 텐서플로 세션을 연다.
    with tf.Session() as sess:
        # 모델을 로드
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        # 폴더가 없으면 만든다.
        if args.output_dir:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

        # 이미지 파일 이름을 가져온다.
        filenames = [
            f.path for f in os.scandir(args.image_dir) if f.is_file() and f.path.endswith(('.jpg'))]

        # 시간을 체크한다.
        start = time.time()
        for f in filenames:
            # 포즈넷에서 이미지를 읽는다,
            input_image, draw_image, output_scale = posenet.read_imgfile(
                f, scale_factor=args.scale_factor, output_stride=output_stride)
            # 텐서플로를 구동한다.
            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )
            # 포즈넷의 값을 가져온다.
            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.25)

            keypoint_coords *= output_scale
            # 아웃풋 디렉토리가 있다면 opencv로 기록한다.
            if args.output_dir:
                draw_image = posenet.draw_skel_and_kp(
                    draw_image, pose_scores, keypoint_scores, keypoint_coords,
                    min_pose_score=0.25, min_part_score=0.25)

                cv2.imwrite(os.path.join(args.output_dir, os.path.relpath(f, args.image_dir)), draw_image)
            # 아웃풋 디렉토리가 없다면 값을 출력한다.
            if not args.notxt:
                print()
                print("Results for image: %s" % f)
                for pi in range(len(pose_scores)):
                    if pose_scores[pi] == 0.:
                        break
                    print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
                    for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                        print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))

        print('Average FPS:', len(filenames) / (time.time() - start))


if __name__ == "__main__":
    main()




