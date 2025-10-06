<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
# import os
# import json
# import argparse
#
# import torch
# from PIL import Image
# from torchvision import transforms
# import matplotlib.pyplot as plt
#
# from vit_model import vit_base_patch16_224_in21k as create_model
#
#
# def main():
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     data_transform = transforms.Compose(
#         [transforms.Resize(256),
#          transforms.CenterCrop(224),
#          transforms.ToTensor(),
#          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
#
#     # load image
#     img_path = r'D:\vision_transformer\flower_photos\daisy\5547758_eea9edfd54_n.jpg'
#     assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
#     img = Image.open(img_path)
#     plt.imshow(img)
#     # [N, C, H, W]
#     img = data_transform(img)
#     # expand batch dimension
#     img = torch.unsqueeze(img, dim=0)
#
#     # read class_indict
#     json_path = './class_indices.json'
#     assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
#
#     with open(json_path, "r") as f:
#         class_indict = json.load(f)
#
#     # create model
#     model = create_model(num_classes=5, has_logits=False).to(device)
#     # load model weights
#     model_weight_path = "./weights/model-9.pth"
#     model.load_state_dict(torch.load(model_weight_path, map_location=device))
#     model.eval()
#     with torch.no_grad():
#         # predict class
#         output = torch.squeeze(model(img.to(device))).cpu()
#         predict = torch.softmax(output, dim=0)
#         predict_cla = torch.argmax(predict).numpy()
#
#     print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
#                                                  predict[predict_cla].numpy())
#     plt.title(print_res)
#     for i in range(len(predict)):
#         print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
#                                                   predict[i].numpy()))
#     plt.show()
#
#
# if __name__ == '__main__':
#     main()


import os
import json
import argparse
from collections import defaultdict
from tqdm import tqdm
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from vit_model import vit_base_patch16_224_in21k as create_model


def predict_image(model, device, img_path, class_indict, show_result=True):
    """预测单张图像"""
    # 图像预处理
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    img = Image.open(img_path)
    img_tensor = data_transform(img).unsqueeze(0).to(device)

    # 推理
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img_tensor)).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    # 结果
    pred_class = class_indict[str(predict_cla)]
    pred_prob = float(predict[predict_cla].numpy())

    # 可视化（可选）
    if show_result:
        plt.imshow(img)
        plt.title(f"Pred: {pred_class} ({pred_prob:.3f})")
        plt.show()

    return pred_class, pred_prob


def predict_folder(model, device, folder_path, class_indict, results_file, true_class=None):
    """预测文件夹内所有图像"""
    # 获取所有图片文件
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

    if not image_files:
        print(f"No images found in {folder_path}")
        return

    # 初始化统计信息
    stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    true_class_name = true_class if true_class else os.path.basename(folder_path)
    true_class_idx = next((k for k, v in class_indict.items() if v == true_class_name), None)

    if true_class_idx is None:
        print(f"Warning: True class '{true_class_name}' not found in class_indict.json")
        return

    # 使用进度条
    for img_file in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(folder_path, img_file)

        try:
            # 执行预测
            pred_class, pred_prob = predict_image(model, device, img_path, class_indict, show_result=False)
            pred_class_idx = next((k for k, v in class_indict.items() if v == pred_class), None)

            # 记录结果（按要求的格式）
            is_correct = int(pred_class == true_class_name)
            with open(results_file, 'a', encoding='utf-8') as f:
                f.write(f"{img_path}|{true_class_idx}|{pred_class_idx}|{pred_prob:.4f}|{is_correct}\n")

            # 更新统计
            stats[true_class_name]['total'] += 1
            if is_correct:
                stats[true_class_name]['correct'] += 1

        except Exception as e:
            print(f"\nError processing {img_file}: {str(e)}")
            continue

    # 返回当前类别的统计信息
    return stats


def predict_big_folder(model, device, big_folder_path, class_indict, results_file):
    """预测大文件夹中的所有子文件夹"""
    subfolders = [f.path for f in os.scandir(big_folder_path) if f.is_dir()]

    if not subfolders:
        print(f"No subfolders found in {big_folder_path}")
        return

    # 初始化总体统计
    all_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    for subfolder in subfolders:
        print(f"\nProcessing folder: {subfolder}")
        folder_stats = predict_folder(model, device, subfolder, class_indict, results_file,
                                      true_class=os.path.basename(subfolder))

        # 合并统计信息
        for cls, data in folder_stats.items():
            all_stats[cls]['correct'] += data['correct']
            all_stats[cls]['total'] += data['total']

    # 写入总体统计（按要求的格式）
    with open(results_file, 'a', encoding='utf-8') as f:
        f.write("\n==== 类别统计 ====\n")
        for cls, data in all_stats.items():
            if data['total'] > 0:
                cls_idx = next((k for k, v in class_indict.items() if v == cls), None)
                if cls_idx is not None:
                    accuracy = data['correct'] / data['total']
                    f.write(f"类别 {cls} ({cls_idx}): 正确数 {data['correct']}/{data['total']} 正确率 {accuracy:.4f}\n")

        # 总体统计
        total_correct = sum(v['correct'] for v in all_stats.values())
        total_images = sum(v['total'] for v in all_stats.values())
        if total_images > 0:
            f.write(f"\n总体正确率: {total_correct / total_images:.4f} ({total_correct}/{total_images})")


def main():
    parser = argparse.ArgumentParser(description='Image Classification Prediction')
    parser.add_argument('--image-path', type=str, help='Path to single image')
    parser.add_argument('--folder-path', type=str, help='Path to folder containing images')
    parser.add_argument('--big-folder-path', type=str, help='Path to big folder containing subfolders')
    parser.add_argument('--true-class', type=str, help='True class label for the images (for accuracy calculation)')
    parser.add_argument('--weights', type=str, default='./weights/model-9.pth', help='Model weights path')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 初始化结果文件
    results_file = "prediction_results.txt"
    if not os.path.exists(results_file):
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("图像路径|真实标签|预测标签|预测概率|是否正确\n")

    # 加载类别标签
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"File '{json_path}' does not exist."
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 关键修改：确保模型类别数与权重匹配
    num_classes = len(class_indict)
    print(f"Initializing model with {num_classes} classes")  # 调试信息
    model = create_model(num_classes=num_classes, has_logits=False).to(device)

    # 加载权重（添加strict=False以允许部分加载）
    try:
        model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)
        print("Weights loaded successfully with strict=False")
    except Exception as e:
        print(f"Error loading weights: {e}")
        # 尝试完全匹配加载
        model = create_model(num_classes=5, has_logits=False).to(device)  # 假设权重是5分类
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print("Weights loaded successfully with 5 classes")

    # 执行预测
    try:
        if args.big_folder_path:
            # 预测大文件夹中的所有子文件夹
            assert os.path.isdir(args.big_folder_path), f"Folder '{args.big_folder_path}' does not exist."
            predict_big_folder(model, device, args.big_folder_path, class_indict, results_file)
        elif args.folder_path:
            # 预测整个文件夹
            assert os.path.isdir(args.folder_path), f"Folder '{args.folder_path}' does not exist."
            predict_folder(model, device, args.folder_path, class_indict, results_file, args.true_class)
        elif args.image_path:
            # 预测单张图片
            assert os.path.isfile(args.image_path), f"Image '{args.image_path}' does not exist."
            pred_class, pred_prob = predict_image(model, device, args.image_path, class_indict)

            # 记录单张图片结果
            with open(results_file, 'a') as f:
                true_class = args.true_class if args.true_class else 'Unknown'
                is_correct = 'Correct' if pred_class == true_class else 'Incorrect'
                f.write(f"{args.image_path}\t{true_class}\t{pred_class}\t{pred_prob:.4f}\t{is_correct}\n")

            # 打印单张图片结果
            print(f"\nPrediction Result:")
            print(f"Image: {args.image_path}")
            print(f"Predicted Class: {pred_class} (Probability: {pred_prob:.4f})")
            if args.true_class:
                print(f"True Class: {args.true_class}")
                print(f"Result: {'Correct' if pred_class == args.true_class else 'Incorrect'}")
        else:
            print("Please specify either --image-path, --folder-path, or --big-folder-path")
            return
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting gracefully.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
# python predict.py --folder-path "D:\vision_transformer\split_datasets\test\bluecar"
#python predict.py --folder-path "D:\vision_transformer\split_datasets\test\greencar"
# python predict.py --folder-path "D:\vision_transformer\split_datasets\test\whitecar"
#python predict.py --folder-path "D:\vision_transformer\split_datasets\test\yellowcar"
# python predict.py --big-folder-path "D:\vision_transformer\split_datasets\test"
#python predict.py --image-path "D:\vision_transformer\data\whitecar\IMG_20250401_115420.jpg"
#python predict.py --big-folder-path "D:\vision_transformer\datasets\test"
=======
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
import tensorflow as tf
import argparse
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 导入自定义模块
from models.TSSC import build_tssc_model
from dataset.data_loader import PeaDiseaseDataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='使用TSSC模型预测豌豆病害')
    parser.add_argument('--image_path', type=str, required=True,
                        help='待预测图像路径')
    parser.add_argument('--weight_path', type=str, default='./weights/best_tssc.h5',
                        help='模型权重文件路径')
    parser.add_argument('--data_dir', type=str, default='./pea_disease_dataset',
                        help='数据集根目录（用于获取类别信息）')
    parser.add_argument('--device', type=str, default='GPU',
                        choices=['GPU', 'CPU'], help='预测设备')
    parser.add_argument('--img_size', type=int, nargs=2, default=[400, 400],
                        help='图像尺寸')
    return parser.parse_args()


def preprocess_image(image_path, img_size):
    """预处理输入图像"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 转换为RGB格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 调整大小
    image = cv2.resize(image, img_size)
    # 归一化
    image = image.astype(np.float32) / 255.0
    # 增加批次维度
    image = np.expand_dims(image, axis=0)
    return image


def predict(args):
    # 设置设备
    if args.device == 'GPU' and tf.test.is_gpu_available():
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("使用GPU进行预测")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("使用CPU进行预测")

    # 获取类别信息
    data_loader = PeaDiseaseDataLoader(
        data_dir=args.data_dir,
        img_size=tuple(args.img_size)
    )
    class_names = data_loader.get_class_names()
    num_classes = len(class_names)

    # 加载模型
    print(f"加载模型权重: {args.weight_path}")
    # 先构建模型结构，再加载权重
    model = build_tssc_model(num_classes=num_classes)
    model.load_weights(args.weight_path)

    # 预处理图像
    print(f"预处理图像: {args.image_path}")
    image = preprocess_image(args.image_path, tuple(args.img_size))

    # 进行预测
    print("进行预测...")
    predictions = model.predict(image)
    pred_probs = predictions[0]
    pred_class_idx = np.argmax(pred_probs)
    pred_class = class_names[pred_class_idx]
    pred_confidence = pred_probs[pred_class_idx] * 100

    # 输出结果
    print("\n预测结果:")
    print(f"输入图像路径: {args.image_path}")
    print(f"预测类别: {pred_class}")
    print(f"置信度: {pred_confidence:.2f}%")

    # 输出前3名预测结果
    print("\nTop 3 预测结果:")
    top3_indices = np.argsort(pred_probs)[-3:][::-1]
    for idx in top3_indices:
        print(f"{class_names[idx]}: {pred_probs[idx] * 100:.2f}%")

    return {
        'class': pred_class,
        'confidence': float(pred_confidence),
        'top3': [(class_names[idx], float(pred_probs[idx] * 100)) for idx in top3_indices]
    }


if __name__ == "__main__":
    args = parse_args()
    predict(args)
>>>>>>> Stashed changes
