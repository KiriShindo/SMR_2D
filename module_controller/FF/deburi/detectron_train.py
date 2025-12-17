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
    train_json        = "C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/seg_dataset_devnet_1module/annotations.json"
    train_images      = "C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/seg_dataset_devnet_1module/"
    output_dir        = "C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/seg_dataset_devnet_1module/prm01"
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
    cfg.MODEL.ROI_HEADS.NUM_CLASSES           = 2

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