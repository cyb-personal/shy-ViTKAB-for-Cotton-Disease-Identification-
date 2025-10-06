<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
import os
import math
import argparse
from datetime import datetime, timedelta
import time
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from my_dataset import MyDataSet
from vit_model import vit_base_patch16_224_in21k as create_model
from utils import read_split_data, train_one_epoch, evaluate
import torch.nn as nn


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 初始化日志和目录
    total_start_time = time.time()
    log_file = "./training_log.txt"
    os.makedirs("./weights", exist_ok=True)

    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write(f"=== Training Started at {datetime.now()} ===\n")
            f.write(f"Config: {args}\n\n")

    tb_writer = SummaryWriter()

    # 数据准备
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    train_dataset = MyDataSet(train_images_path, train_images_label, data_transform["train"])
    val_dataset = MyDataSet(val_images_path, val_images_label, data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'Using {nw} dataloader workers every process')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=4, collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=nw, collate_fn=val_dataset.collate_fn)

    # 模型初始化（关键修改）
    model = create_model(
        num_classes=args.num_classes,
        has_logits=False,
        use_kan=args.use_kan,
        use_biformer=args.use_biformer
    ).to(device)
#-------------------------------------
    # 参数统计
    print("\n===== 参数统计 =====")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数: {trainable_params / 1e6:.2f}M ({trainable_params / total_params:.1%})")

    # 按模块统计
    print("\n各模块参数量:")
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        print(f"{name:15s}: {module_params / 1e6:.2f}M")

    # 启用调试模式
    model.set_debug(True)  # 确保这里可以调用
#------------------------------------------
    # 冻结逻辑
    if args.freeze_kan or args.freeze_all:
        for name, param in model.named_parameters():
            if 'kan' in name.lower():  # 冻结所有KAN参数
                param.requires_grad_(False)
                print(f"冻结KAN层: {name}")

    if args.freeze_biformer or args.freeze_all:
        for name, param in model.named_parameters():
            if 'biformer' in name.lower():  # 冻结所有BiFormer参数
                param.requires_grad_(False)
                print(f"冻结BiFormer层: {name}")

    if args.freeze_all:
        for name, param in model.named_parameters():
            if 'head' not in name:  # 冻结所有非分类层
                param.requires_grad_(False)
                print(f"冻结非分类层: {name}")

       #替换原有的激活KAN检测代码
        if args.use_kan:
            active_kan_layers = [
                name for name, param in model.named_parameters()
                if 'kan' in name.lower() and param.requires_grad
            ]
            print("✅ 实际可训练的KAN层：" if active_kan_layers else "❌ 所有KAN层已被冻结")
            for name in active_kan_layers:
                print(f"  - {name}")
        kan_params = sum(p.numel() for n, p in model.named_parameters() if 'kan' in n)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"KAN参数量: {kan_params}/{total_params} ({kan_params / total_params:.1%})")

    # 在train.py中添加临时检查
    print(model.blocks[0].attn)  # 应输出: BiFormerAttention(...)

    # 权重加载（改进版）
    if args.weights != "":
        weights_dict = torch.load(args.weights, map_location=device)

        # 动态删除冲突键
        del_keys = [k for k in weights_dict.keys()
                    if 'head' in k or 'pre_logits' in k]
        for k in del_keys:
            weights_dict.pop(k, None)

        # 加载权重并初始化新增的KAN参数
        model.load_state_dict(weights_dict, strict=False)

        if args.use_kan:
            for name, param in model.named_parameters():
                if 'kan' in name.lower() and name not in weights_dict:
                    if 'weight' in name or 'coeff' in name:
                        if param.dim() >= 2:  # 检查张量维度是否 >= 2
                            nn.init.kaiming_normal_(param, mode='fan_in')
                    elif 'bias' in name:
                        nn.init.zeros_(param)
                    print(f"初始化KAN参数：{name}")

    # 冻结设置（修正版）
    if args.freeze_layers:
        print("\n===== 冻结参数检查 =====")
        frozen_params = []
        for name, param in model.named_parameters():
            # 不冻结：1)分类头 2)KAN层 3)显式要求不冻结的层
            freeze = (
                    ("head" not in name) and
                    ("pre_logits" not in name) and
                    ("kan" not in name.lower())
            )
            param.requires_grad_(not freeze)
            if freeze:
                frozen_params.append(name)

        print(f"共冻结 {len(frozen_params)} 个参数")
        print("示例冻结参数:", frozen_params[:5])
    else:
        print("\n===== 所有参数可训练 =====")
        for param in model.parameters():
            param.requires_grad_(True)

    # 在 train.py 的 main() 函数中添加
    if args.freeze_kan or args.freeze_all:
        for name, param in model.named_parameters():
            if any(k in name.lower() for k in ['kan', 'mlp.kan']):  # 兼容不同命名
                param.requires_grad_(False)
                print(f"冻结KAN参数: {name}") if args.verbose else None

    if args.freeze_biformer or args.freeze_all:
        for name, param in model.named_parameters():
            if 'biformer' in name.lower() or 'attn.qkv' in name:  # 覆盖BiFormer相关层
                param.requires_grad_(False)
                print(f"冻结BiFormer参数: {name}") if args.verbose else None


    # 替换原有统计代码
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_kan_params = sum(
        p.numel() for n, p in model.named_parameters()
        if 'kan' in n.lower() and not p.requires_grad
    )

    print(f"\n总参数量: {total_params}")
    print(f"可训练参数: {trainable_params} ({trainable_params / total_params:.1%})")
    print(f"冻结的KAN参数: {frozen_kan_params} ({frozen_kan_params / total_params:.1%})")

    # 优化器和学习率调度器
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        pg,
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.05,
        eps=1e-8
    )
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    # 训练循环
    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        # 训练和验证
        train_loss, train_acc = train_one_epoch(
            model, optimizer, train_loader, device, epoch)

        scheduler.step()

        val_loss, val_acc = evaluate(
            model, val_loader, device, epoch)

        # 记录日志
        epoch_duration = timedelta(seconds=time.time() - epoch_start_time)
        with open(log_file, "a") as f:
            f.write(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"Train: loss={train_loss:.4f}, acc={train_acc:.4f} | "
                f"Val: loss={val_loss:.4f}, acc={val_acc:.4f} | "
                f"Duration: {str(epoch_duration)}\n"
            )

        # TensorBoard记录
        tb_writer.add_scalar("train_loss", train_loss, epoch)
        tb_writer.add_scalar("train_acc", train_acc, epoch)
        tb_writer.add_scalar("val_loss", val_loss, epoch)
        tb_writer.add_scalar("val_acc", val_acc, epoch)
        tb_writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

        # 分层监控
        depth = len(model.blocks)
        for name, param in model.named_parameters():
            if 'blocks.' in name:
                layer_idx = int(name.split('.')[1])  # 获取层索引
                if layer_idx < depth // 3:
                    tb_writer.add_histogram(f'early_layers/{name}', param, epoch)
                elif layer_idx >= 2 * depth // 3:
                    tb_writer.add_histogram(f'late_layers/{name}', param, epoch)

        # 保存模型
        torch.save(model.state_dict(), f"./weights/model-{epoch}.pth")

    # 训练结束
    total_duration = timedelta(seconds=int(time.time() - total_start_time))
    with open(log_file, "a") as f:
        f.write(f"\n=== Training Completed ===\n")
        f.write(f"Total epochs: {args.epochs}\n")
        f.write(f"Total duration: {total_duration}\n")
        f.write(f"Avg time per epoch: {total_duration / args.epochs}\n")

    print(f"\nTraining finished. Total time: {total_duration}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 基础参数
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)  #8
    parser.add_argument('--lr', type=float, default=0.001)    #0.01
    parser.add_argument('--lrf', type=float, default=0.01)

    parser.add_argument('--use-biformer', action='store_true', help='启用BiFormer注意力')
    # KAN相关参数（新增）
    parser.add_argument('--use-kan', action='store_true', help='启用KAN层替代MLP')
    parser.add_argument('--freeze-kan', action='store_true', help='冻结KAN层参数')

    # 在 argparse 部分新增以下参数
    parser.add_argument('--freeze-biformer', action='store_true', help='冻结所有BiFormer参数')
    parser.add_argument('--freeze-all', action='store_true', help='冻结所有非分类层（包括KAN和BiFormer）')
    # 路径参数
    parser.add_argument('--data-path', type=str,
                        default=r"D:\vision_transformer\datasets\train")
    parser.add_argument('--weights', type=str,
                        default=r'D:\vision_transformer\vit_base_patch16_224_in21k.pth')
    parser.add_argument('--verbose', action='store_true', help='显示详细冻结信息')
    # 训练控制
    parser.add_argument('--freeze-layers', action='store_true', help='是否冻结非分类层')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='梯度裁剪阈值')
    parser.add_argument('--drop-path', type=float, default=0.0, help='随机深度概率')
    parser.add_argument('--warmup-epochs', type=int, default=0, help='学习率预热epoch数')
    opt = parser.parse_args()
    main(opt)

 # # 完全解冻模式（所有参数可训练）
# python train.py --use-kan
# # 部分冻结模式（仅 KAN 和分类头可训练）
# python train.py --use-kan --freeze-layers
#启用Biformer
#python train.py --use-biformer
# # 同时启用 BiFormer 和 KAN
# python train.py --use-biformer --use-kan --num_classes 4 --data-path ./split_dataset
#冻结KAN
#python train.py --use-kan --use-biformer --freeze-kan --num_classes 4 --data-path ./split_dataset --batch-size 8 --epochs 30
#全部冻结
# python train.py --use-kan --use-biformer --freeze-kan --freeze-biformer --num_classes 4 --data-path D:\vision_transformer\datasets\train --batch-size 8 --epochs 10
#python train.py --use-kan --num_classes 4 --data-path ./split_dataset
=======
import tensorflow as tf
import argparse
import os
import datetime
from tensorflow.keras import optimizers, metrics, callbacks
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.metrics import f1_score, classification_report
import numpy as np

# 导入自定义模块
from models.TSSC import build_tssc_model
from dataset.data_loader import PeaDiseaseDataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='训练TSSC豌豆病害识别模型')
    parser.add_argument('--data_dir', type=str, default='./pea_disease_dataset',
                        help='数据集根目录')
    parser.add_argument('--epochs', type=int, default=60,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='权重衰减系数')
    parser.add_argument('--save_dir', type=str, default='./weights',
                        help='模型权重保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='日志保存目录')
    parser.add_argument('--device', type=str, default='GPU',
                        choices=['GPU', 'CPU'], help='训练设备')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--test_split', type=float, default=0.2,
                        help='测试集比例')
    parser.add_argument('--img_size', type=int, nargs=2, default=[400, 400],
                        help='图像尺寸')
    return parser.parse_args()


def main():
    args = parse_args()

    # 设置设备
    if args.device == 'GPU' and tf.test.is_gpu_available():
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("使用GPU进行训练")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("使用CPU进行训练")

=======
import tensorflow as tf
import argparse
import os
import datetime
from tensorflow.keras import optimizers, metrics, callbacks
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.metrics import f1_score, classification_report
import numpy as np

# 导入自定义模块
from models.TSSC import build_tssc_model
from dataset.data_loader import PeaDiseaseDataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='训练TSSC豌豆病害识别模型')
    parser.add_argument('--data_dir', type=str, default='./pea_disease_dataset',
                        help='数据集根目录')
    parser.add_argument('--epochs', type=int, default=60,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='权重衰减系数')
    parser.add_argument('--save_dir', type=str, default='./weights',
                        help='模型权重保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='日志保存目录')
    parser.add_argument('--device', type=str, default='GPU',
                        choices=['GPU', 'CPU'], help='训练设备')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--test_split', type=float, default=0.2,
                        help='测试集比例')
    parser.add_argument('--img_size', type=int, nargs=2, default=[400, 400],
                        help='图像尺寸')
    return parser.parse_args()


def main():
    args = parse_args()

    # 设置设备
    if args.device == 'GPU' and tf.test.is_gpu_available():
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("使用GPU进行训练")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("使用CPU进行训练")

>>>>>>> Stashed changes
=======
import tensorflow as tf
import argparse
import os
import datetime
from tensorflow.keras import optimizers, metrics, callbacks
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.metrics import f1_score, classification_report
import numpy as np

# 导入自定义模块
from models.TSSC import build_tssc_model
from dataset.data_loader import PeaDiseaseDataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='训练TSSC豌豆病害识别模型')
    parser.add_argument('--data_dir', type=str, default='./pea_disease_dataset',
                        help='数据集根目录')
    parser.add_argument('--epochs', type=int, default=60,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='权重衰减系数')
    parser.add_argument('--save_dir', type=str, default='./weights',
                        help='模型权重保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='日志保存目录')
    parser.add_argument('--device', type=str, default='GPU',
                        choices=['GPU', 'CPU'], help='训练设备')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--test_split', type=float, default=0.2,
                        help='测试集比例')
    parser.add_argument('--img_size', type=int, nargs=2, default=[400, 400],
                        help='图像尺寸')
    return parser.parse_args()


def main():
    args = parse_args()

    # 设置设备
    if args.device == 'GPU' and tf.test.is_gpu_available():
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("使用GPU进行训练")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("使用CPU进行训练")

>>>>>>> Stashed changes
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 加载数据
    print("加载数据集...")
    data_loader = PeaDiseaseDataLoader(
        data_dir=args.data_dir,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size
    )
    train_ds, val_ds, test_ds = data_loader.get_datasets(
        val_split=args.val_split,
        test_split=args.test_split
    )
    num_classes = data_loader.num_classes
    print(f"类别数量: {num_classes}, 类别名称: {data_loader.get_class_names()}")

    # 构建模型
    print("构建TSSC模型...")
    model = build_tssc_model(num_classes=num_classes)
    model.summary()

    # 定义优化器
    optimizer = optimizers.Adam(
        learning_rate=args.lr,
        weight_decay=args.weight_decay
    )

    # 编译模型
    model.compile(
        optimizer=optimizer,
        loss=CategoricalCrossentropy(from_logits=False),
        metrics=[
            metrics.CategoricalAccuracy(name='accuracy'),
            metrics.Precision(name='precision'),
            metrics.Recall(name='recall')
        ]
    )

    # 定义回调函数
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = callbacks.TensorBoard(
        log_dir=os.path.join(args.log_dir, current_time),
        histogram_freq=1
    )

    # 模型保存回调（保存验证集性能最好的模型）
    model_checkpoint = callbacks.ModelCheckpoint(
        filepath=os.path.join(args.save_dir, 'best_tssc.h5'),
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    # 学习率衰减
    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )

    # 早停策略
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    # 训练模型
    print("开始训练...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[
            tensorboard_callback,
            model_checkpoint,
            lr_scheduler,
            early_stopping
        ]
    )

    # 在测试集上评估
    print("在测试集上评估模型...")
    test_loss, test_acc, test_precision, test_recall = model.evaluate(test_ds, verbose=1)

    # 计算F1分数
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    test_f1 = f1_score(y_true, y_pred, average='macro')

    print(f"测试集结果:")
    print(f"损失: {test_loss:.4f}")
    print(f"准确率: {test_acc:.4f}")
    print(f"精确率: {test_precision:.4f}")
    print(f"召回率: {test_recall:.4f}")
    print(f"宏平均F1: {test_f1:.4f}")

    # 打印详细分类报告
    print("\n分类报告:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=data_loader.get_class_names()
    ))

    # 保存最终模型
    final_model_path = os.path.join(args.save_dir, 'final_tssc.h5')
    model.save(final_model_path)
    print(f"最终模型已保存至: {final_model_path}")


if __name__ == "__main__":
    main()
<<<<<<< Updated upstream
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
