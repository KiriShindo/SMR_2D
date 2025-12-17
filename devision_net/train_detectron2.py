# #!/usr/bin/env python3
# # train_maskrcnn_with_progress.py

# import os
# import math
# import torch
# from tqdm import tqdm
# from detectron2.utils.logger import setup_logger
# setup_logger()

# print("CUDA available:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("CUDA device count:", torch.cuda.device_count())
#     print("  Device 0:", torch.cuda.get_device_name(0))

# from detectron2 import model_zoo
# from detectron2.config import get_cfg
# import os
# from detectron2.data.datasets import register_coco_instances
# from detectron2.engine import DefaultTrainer, HookBase
# from detectron2.data import DatasetCatalog

# class ProgressBarHook(HookBase):
#     def before_train(self):
#         total = self.trainer.cfg.SOLVER.MAX_ITER
#         self._tqdm = tqdm(total=total, desc="Train Iter", leave=False)

#     def after_step(self):
#         self._tqdm.update(1)

# class CheckpointEveryNIter(HookBase):
#     def __init__(self, save_iter, output_dir):
#         self.save_iter = save_iter
#         self.output_dir = output_dir

#     def after_step(self):
#         it = self.trainer.iter
#         if it % self.save_iter == 0 and it > 0:
#             ckpt_name = f"model_iter_{it:07d}"
#             self.trainer.checkpointer.save(
#                 ckpt_name,
#                 **{"iteration": it}
#             )

# def main():
#     # ——— ユーザ設定 ———
#     train_json = "C:/Users/akami/HangngChainRobot/segmentaion_dataset/dataset01/annotations.json"
#     train_images = "C:/Users/akami/HangngChainRobot/segmentaion_dataset/dataset01/"
#     output_dir = "C:/Users/akami/HangngChainRobot/segmentaion_dataset/dataset01/prm01"  # ← 保存先フォルダ
#     save_every_n_iter = 100    # 10イテレーションごとに保存
#     num_epochs = 1000          # 100エポック回す
#     # ——————————————————

#     # 1) データセット登録
#     register_coco_instances("Train", {}, train_json, train_images)
#     cfg = get_cfg()
#     cfg.merge_from_file(
#         model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
#     )
#     cfg.DATASETS.TRAIN = ("Train",)
#     cfg.DATASETS.TEST  = ()              # 検証データなし
#     cfg.DATALOADER.NUM_WORKERS = 4

#     # バッチサイズ・イテレーション設定
#     cfg.SOLVER.IMS_PER_BATCH = 8

#     # データセット長から「1エポックあたりのイテレーション数」を計算
#     # Detectron2 の DataLoader は「1 iter = 1 バッチ処理」です
#     dataset_dicts = DatasetCatalog.get("Train")
#     num_samples = len(dataset_dicts)
#     iter_per_epoch = math.ceil(num_samples / cfg.SOLVER.IMS_PER_BATCH)
#     print(f"Samples: {num_samples}, iters/epoch: {iter_per_epoch}")

#     # 3) Solver／ROIヘッド設定
#     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
#         "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
#     )
#     cfg.SOLVER.BASE_LR      = 0.1
#     # 100エポック分の総イテレーション数を設定
#     cfg.SOLVER.MAX_ITER     = num_epochs
#     # チェックポイント保存間隔（デフォルト5000→10に変更）
#     cfg.SOLVER.CHECKPOINT_PERIOD = save_every_n_iter
#     cfg.SOLVER.STEPS       = []
#     cfg.TEST.EVAL_PERIOD   = 0      # 評価スキップ
#     cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES          = 6

#     # 出力先ディレクトリ
#     cfg.OUTPUT_DIR = output_dir
#     os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

#     # 4) GPU を使う
#     cfg.MODEL.DEVICE = "cuda"

#     # Trainer 準備
#     trainer = DefaultTrainer(cfg)
#     # カスタム Hook を登録
#     trainer.register_hooks([
#         ProgressBarHook(),
#         CheckpointEveryNIter(save_every_n_iter, cfg.OUTPUT_DIR)
#     ])
#     trainer.resume_or_load(resume=False)
#     trainer.train()

# if __name__ == "__main__":
#     main()



### 学習曲線の追加
#!/usr/bin/env python3
# train_maskrcnn_with_loss_curve.py

import os
import math
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from detectron2.utils.logger import setup_logger

# Logger setup
setup_logger()
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  CUDA device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"    Device {i}:", torch.cuda.get_device_name(i))

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog
from detectron2.engine import DefaultTrainer, HookBase

class ProgressBarHook(HookBase):
    """TQDM で train の進捗バーを表示"""
    def before_train(self):
        total = self.trainer.cfg.SOLVER.MAX_ITER
        self._tqdm = tqdm(total=total, desc="Train Iter", leave=False)
    def after_step(self):
        self._tqdm.update(1)

class CheckpointEveryNIter(HookBase):
    """N イテレーションごとにモデル重みを保存"""
    def __init__(self, save_iter, output_dir):
        self.save_iter  = save_iter
        self.output_dir = output_dir
    def after_step(self):
        it = self.trainer.iter
        if it > 0 and it % self.save_iter == 0:
            name = f"model_iter_{it:07d}"
            self.trainer.checkpointer.save(name, **{"iteration": it})
            print(f"[Checkpoint] saved {name}")

class SaveLossCurveHook(HookBase):
    """
    N イテレーションごとに total_loss のみをプロット・保存する Hook
    """
    def __init__(self, save_iter, output_dir):
        self.save_iter  = save_iter
        self.output_dir = output_dir

    def after_step(self):
        it = self.trainer.iter
        if it > 0 and it % self.save_iter == 0:
            # total_loss の履歴を取得
            history_buffer = self.trainer.storage.history("total_loss")
            losses = history_buffer.values()
            train_losses, _ = zip(*losses)
            iters  = list(range(1, len(losses) + 1))

            # プロット（損失値のみ一本線）
            plt.figure()
            plt.plot(iters, train_losses)
            plt.xlabel("Iteration")
            plt.ylabel("Total Loss")
            plt.title(f"Training Loss Curve (iter {it})")
            plt.grid(True)

            # 保存
            fname = f"loss_curve_iter_{it:07d}.png"
            path  = os.path.join(self.output_dir, fname)
            plt.savefig(path)
            plt.close()
            print(f"[LossCurve] saved {fname}")

def main():
    # —— ユーザ設定 —— 
    train_json        = "C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devision_net/devnet_data/all_dataset/annotations.json"
    train_images      = "C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devision_net/devnet_data/all_dataset/"
    output_dir        = "C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devision_net/devnet_data/all_dataset/prm01"
    save_every_n_iter = 100      # 100 iter ごと
    max_iter          = 3001    # 合計 iter
    # ————————————

    # 1) データセット登録
    register_coco_instances("Train", {}, train_json, train_images)

    # 2) cfg 構築
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.DATASETS.TRAIN = ("Train",)
    cfg.DATASETS.TEST  = ()
    cfg.DATALOADER.NUM_WORKERS = 2

    # 3) Solver・モデルヘッド設定
    cfg.SOLVER.IMS_PER_BATCH                  = 2
    cfg.SOLVER.BASE_LR                        = 0.00025
    cfg.SOLVER.MAX_ITER                       = max_iter
    cfg.SOLVER.CHECKPOINT_PERIOD              = save_every_n_iter
    cfg.SOLVER.STEPS                          = []
    cfg.TEST.EVAL_PERIOD                      = 0
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE  = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES           = 1

    # 4) 重み初期化 & 出力先
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.OUTPUT_DIR     = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # 5) デバイス
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 6) Trainer & Hook 登録
    trainer = DefaultTrainer(cfg)
    trainer.register_hooks([
        ProgressBarHook(),
        CheckpointEveryNIter(save_every_n_iter, cfg.OUTPUT_DIR),
        SaveLossCurveHook(save_every_n_iter, cfg.OUTPUT_DIR),
    ])
    trainer.resume_or_load(resume=False)

    # 7) 学習開始
    trainer.train()

if __name__ == "__main__":
    main()





# import os
# import torch
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from detectron2.utils.logger import setup_logger

# from detectron2 import model_zoo
# from detectron2.config import get_cfg
# from detectron2.data.datasets import register_coco_instances
# from detectron2.engine import DefaultTrainer, HookBase
# from detectron2.data import (
#     get_detection_dataset_dicts,
#     DatasetMapper,
#     build_detection_train_loader,
# )

# # Logger setup
# setup_logger()
# print("CUDA available:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("  CUDA device count:", torch.cuda.device_count())
#     for i in range(torch.cuda.device_count()):
#         print(f"    Device {i}:", torch.cuda.get_device_name(i))

# class ProgressBarHook(HookBase):
#     """TQDM で train の進捗バーを表示"""
#     def before_train(self):
#         total = self.trainer.cfg.SOLVER.MAX_ITER
#         self._tqdm = tqdm(total=total, desc="Train Iter", leave=False)
#     def after_step(self):
#         self._tqdm.update(1)

# class CheckpointEveryNIter(HookBase):
#     """N イテレーションごとにモデル重みを保存"""
#     def __init__(self, save_iter, output_dir):
#         self.save_iter  = save_iter
#         self.output_dir = output_dir
#     def after_step(self):
#         it = self.trainer.iter
#         if it > 0 and it % self.save_iter == 0:
#             name = f"model_iter_{it:07d}"
#             self.trainer.checkpointer.save(name, **{"iteration": it})
#             print(f"[Checkpoint] saved {name}")

# class ValidationLossHook(HookBase):
#     """
#     EVAL_PERIOD ごとに検証データで平均損失を計算し、
#     'validation_loss' として storage に保存する Hook
#     """
#     def __init__(self, eval_period):
#         self.eval_period = eval_period

#     def after_step(self):
#         next_iter = self.trainer.iter + 1
#         if next_iter % self.eval_period == 0:
#             cfg = self.trainer.cfg

#             # 1) 検証用データの dict リストを取得
#             dataset_dicts = get_detection_dataset_dicts(cfg.DATASETS.TEST)
#             # 2) mapper: 画像読み込み＋前処理
#             mapper = DatasetMapper(cfg, is_train=False)
#             # 3) DataLoader を構築
#             val_loader = build_detection_train_loader(
#                 dataset_dicts,
#                 mapper=mapper,
#                 total_batch_size=cfg.SOLVER.IMS_PER_BATCH,
#                 num_workers=cfg.DATALOADER.NUM_WORKERS,
#             )  # :contentReference[oaicite:1]{index=1}

#             total_loss = 0.0
#             count = 0
#             # train モード & 勾配計算オフ
#             self.trainer.model.train()
#             with torch.no_grad():
#                 for batch in val_loader:
#                     loss_dict = self.trainer.model(batch)
#                     batch_loss = sum(loss for loss in loss_dict.values())
#                     total_loss += batch_loss.item()
#                     count += 1

#             avg_loss = total_loss / max(count, 1)
#             self.trainer.storage.put_scalar("validation_loss", avg_loss)
#             print(f"[Validation] iter {next_iter}: avg_loss={avg_loss:.4f}")

# class SaveLossCurveHook(HookBase):
#     """
#     N イテレーションごとに
#     - 訓練時の total_loss
#     - 検証時の validation_loss
#     を同一グラフにプロット・保存する Hook
#     """
#     def __init__(self, save_iter, output_dir):
#         self.save_iter  = save_iter
#         self.output_dir = output_dir

#     def after_step(self):
#         it = self.trainer.iter
#         if it > 0 and it % self.save_iter == 0:
#             storage = self.trainer.storage

#             # 訓練損失履歴
#             train_hist = storage.history("total_loss").values()
#             train_losses, _ = zip(*train_hist)

#             # 検証損失履歴
#             val_hist = storage.history("validation_loss").values()
#             if val_hist:
#                 val_losses, _ = zip(*val_hist)
#                 val_iters = [self.save_iter * i for i in range(1, len(val_losses) + 1)]
#             else:
#                 val_losses = []
#                 val_iters = []

#             train_iters = list(range(1, len(train_losses) + 1))

#             plt.figure()
#             plt.plot(train_iters, train_losses, label="Train Loss")
#             if val_losses:
#                 plt.plot(val_iters, val_losses,   label="Eval Loss")
#             plt.xlabel("Iteration")
#             plt.ylabel("Loss")
#             plt.title(f"Loss Curve @ iter {it}")
#             plt.grid(True)
#             plt.legend()

#             fname = f"loss_curve_with_eval_iter_{it:07d}.png"
#             path  = os.path.join(self.output_dir, fname)
#             plt.savefig(path)
#             plt.close()
#             print(f"[LossCurve] saved {fname}")

# def main():
#     # —— ユーザ設定 ——
#     train_json        = "C:/Users/akami/HangngChainRobot/segmentaion_dataset/dataset_train/annotations.json"
#     train_images      = "C:/Users/akami/HangngChainRobot/segmentaion_dataset/dataset_train/"
#     eval_json         = "C:/Users/akami/HangngChainRobot/segmentaion_dataset/dataset_eval/annotations.json"
#     eval_images       = "C:/Users/akami/HangngChainRobot/segmentaion_dataset/dataset_eval/"
#     output_dir        = "C:/Users/akami/HangngChainRobot/segmentaion_dataset/train_result/prm01"
#     save_every_n_iter = 100      # 評価・保存の間隔
#     max_iter          = 1001     # 合計イテレーション
#     # ————————————

#     # データセット登録
#     register_coco_instances("Train", {}, train_json, train_images)
#     register_coco_instances("Eval",  {}, eval_json,  eval_images)

#     # cfg 設定
#     cfg = get_cfg()
#     cfg.merge_from_file(
#         model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
#     )
#     cfg.DATASETS.TRAIN         = ("Train",)
#     cfg.DATASETS.TEST          = ("Eval",)
#     cfg.DATALOADER.NUM_WORKERS = 4

#     # Solver 設定
#     cfg.SOLVER.IMS_PER_BATCH                 = 8
#     cfg.SOLVER.BASE_LR                       = 0.1
#     cfg.SOLVER.MAX_ITER                      = max_iter
#     cfg.SOLVER.CHECKPOINT_PERIOD             = save_every_n_iter
#     cfg.SOLVER.STEPS                         = []
#     cfg.TEST.EVAL_PERIOD                     = save_every_n_iter
#     cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES          = 6

#     # 初期重み & 出力先
#     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
#         "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
#     )
#     cfg.OUTPUT_DIR = output_dir
#     os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

#     # デバイス
#     cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#     # Trainer & Hook 登録
#     trainer = DefaultTrainer(cfg)
#     trainer.register_hooks([
#         ProgressBarHook(),
#         CheckpointEveryNIter(save_every_n_iter, cfg.OUTPUT_DIR),
#         ValidationLossHook(save_every_n_iter),
#         SaveLossCurveHook(save_every_n_iter, cfg.OUTPUT_DIR),
#     ])
#     trainer.resume_or_load(resume=False)

#     # 学習開始
#     trainer.train()

# if __name__ == "__main__":
#     main()
