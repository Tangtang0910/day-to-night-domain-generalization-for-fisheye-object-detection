# Day-to-Night Fisheye Object Detection via Image Translation and Self-Training

本專案研究如何提升魚眼物件偵測模型在日間資料訓練、夜間資料測試情境下的泛化能力。

研究以 FishEye8K dataset 為基礎，使用 YOLOv8s 進行物件偵測，並比較傳統影像處理、GAN 影像轉換與 Self-training 等方法對夜間物件偵測表現的影響。

## 研究目標

1. 使用日間資料訓練物件偵測模型。
2. 評估模型在夜間資料上的 domain generalization 能力。
3. 透過影像前處理模擬夜間影像。
4. 使用 GAN 進行日間到夜間的影像轉換。
5. 使用 pseudo-label 與 self-training 改善模型在夜間資料上的表現。
6. 比較不同方法在 Precision、Recall、mAP50 與 mAP50-95 上的差異。

## 實驗前提

- FishEye8K dataset 中，M(morning)+A(afternoon) 部分包含多種 camera 的資料，而 N(night) 部分只有 camera 3 與 camera 4 的資料。為了降低實驗範圍與額外變因，本研究只使用：M+A dataset 的 camera 3、camera 4 作為訓練資料。
  
  因此，本實驗主要評估模型從日間資料泛化到夜間資料的能力，而非不同 camera domain 之間的泛化能力。

- 為了進行 self-training，本研究的 N dataset 會以 1:1 比例切分為 Pseudo-label subset 與 Evaluation subset(每個 camera 內部 1:1 分配)。

  - **Pseudo-label subset**：只使用影像產生 pseudo-label，並加入 self-training。
  - **Evaluation subset**：完全不參與 pseudo-label 產生或模型訓練，只用於最終測試。

  為確保不同方法之間可以公平比較，資料切分使用固定的 random seed，所有 self-training 實驗均使用相同的資料切分方式與相同的 Evaluation subset，以確保所有實驗使用相同且可重現的資料分割。最終報告的 Precision、Recall、mAP50 與 mAP50-95 皆以 Evaluation subset 計算。

- 所有實驗均使用相同的 YOLOv8s 預訓練模型，以確保不同方法之間具有可比較性。

## 使用的資料集與模型

### Dataset

本研究使用 FishEye8K 魚眼影像資料集。

官方 GitHub repository: https://github.com/MoyoG/FishEye8K

Hugging Face dataset: https://huggingface.co/datasets/Voxel51/fisheye8k

### Object Detection Model

所有物件偵測實驗均使用：

- Model: YOLOv8s
- Initial weights: `yolov8s.pt`
- Framework: Ultralytics YOLO

Ultralytics repository: https://github.com/ultralytics/ultralytics

### GAN Model

影像日夜轉換使用 img2img-turbo GAN model。

Repository: https://github.com/GaParmar/img2img-turbo

## 評估指標

- Precision
- Recall
- mAP50
- mAP50-95

本研究以 Evaluation subset 的 mAP50-95 作為主要模型比較指標。

## 實驗方法

### - Baseline

直接使用原始日間資料訓練 YOLOv8s，未進行額外的夜間風格轉換或 self-training。

### - Method 1：CV Preprocessing

使用傳統影像處理方法對日間影像進行夜間風格模擬。

控制的影像變數包括：

| 變數 | 效果 | 實驗係數 |
|---|---|---|
| Brightness | 調整亮度 | 0.85、0.60、0.45 |
| Contrast | 調整對比度 | 0.80、0.60、0.40 |
| Saturation | 調整飽和度 | 0.00、0.30、0.50、0.70 |
| Gamma | 調整 gamma | 1.30、1.50、1.70 |
| Vignette | 模擬邊緣變暗 | 1.00、1.50、2.00 |
| Noise | 模擬感測器雜訊 | sigma = 5、10、15 |
| Blur | 模擬模糊 | radius = 0.5、1.0、1.5 |

本研究暫不考慮 color temperature，原因是夜間影像主要呈現灰、黑、白等低彩度特徵。

此外，也測試多種影像變數組合：

```text
Brightness = 0.85
Contrast = 0.40
Saturation = 0.00
Vignette = 2.00
Noise sigma = 5
Blur radius = 1.0
```

### - Method 2：GAN Preprocessing

使用 GAN 將日間影像轉換為夜間風格，再使用轉換後的影像進行訓練。

### - Method 3：GAN + CV Preprocessing

先使用 GAN 進行夜間風格轉換，再使用 CV preprocessing 將影像的 saturation 設定為 0.00，將轉換完畢的影像進行訓練。

### - Method 4：Self-training

先使用初始模型對部分夜間影像產生 pseudo-label，再依照 confidence threshold 篩選可信的 pseudo-label，加入訓練資料進行 self-training。

測試的 confidence threshold 為 `0.6`、`0.7`、`0.8` 與 `0.9`。

### - Method 5：GAN Preprocessing + Self-training

先使用 GAN 進行日間到夜間的影像轉換，再使用 self-training。

測試的 confidence threshold 為 `0.4`、`0.5`、`0.6`、`0.7` 與 `0.8`。

### - Method 6：CV Preprocessing + Self-training

根據 Method 1 的實驗結果，saturation = `0.00` 的 CV preprocessing 設定在 Evaluation subset 上表現最佳，因此本方法採用相同的設定，再結合 self-training。

測試的 confidence threshold 為 `0.4`、`0.5`、`0.6`、`0.7` 與 `0.8`。

### - Method 7：GAN + CV Preprocessing + Self-training

先使用 GAN 進行夜間風格轉換，再將影像 saturation 設定為 0.00，最後使用 self-training。

測試的 confidence threshold 為 `0.4`、`0.5`、`0.6` 與 `0.7`。


## 主要實驗結果

結果皆為 Evaluation subset 結果。Baseline 與非 self-training 方法也使用相同的 Evaluation subset 評估。

| Method | Best setting | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| Baseline | None | 0.4841 | 0.0981 | 0.0995 | 0.0483 |
| Method 1: CV | Saturation = 0.00 | 0.3815 | 0.1473 | 0.1558 | 0.0766 |
| Method 2: GAN | GAN preprocessing | 0.4110 | 0.1530 | 0.1600 | 0.0780 |
| Method 3: GAN + CV | Best setting | 0.3518 | 0.1783 | 0.1761 | 0.0845 |
| Method 4: Self-training | Threshold = 0.7 | 0.1701 | 0.1871 | 0.1162 | 0.0597 |
| Method 5: GAN + Self-training | Threshold = 0.5 | 0.3846 | 0.1901 | 0.1785 | 0.0875 |
| Method 6: CV + Self-training | Threshold = 0.6 | 0.5680 | 0.1777 | 0.1940 | 0.0969 |
| Method 7: GAN + CV + Self-training | Threshold = 0.6 | 0.3623 | 0.1977 | 0.1944 | **0.1007** |

## 最佳模型

以 Evaluation subset 的 mAP50-95 作為主要評估指標，最佳模型為 **Method 7，threshold = 0.6**：

```text
CV Preprocessing (saturation = 0.00) + GAN + Self-training
```

| 指標 | Test 結果 |
|---|---:|
| Precision | 0.3623 |
| Recall | 0.1977 |
| mAP50 | 0.1944 |
| mAP50-95 | **0.1007** |

與 Baseline 比較：

| 指標 | Baseline | Best Model | 改善 |
|---|---:|---:|---:|
| Precision | 0.4841 | 0.3623 | -25.2% |
| Recall | 0.0981 | 0.1977 | +101.6% |
| mAP50 | 0.0995 | 0.1944 | +95.5% |
| mAP50-95 | 0.0483 | 0.1007 | **+108.7%** |

最佳模型在 Precision 上低於 Baseline，但在 Recall、mAP50 與 mAP50-95 上均明顯提升。這表示最佳模型能偵測到更多夜間目標，且整體定位表現較好，但同時也產生較多誤報。

## 不同指標的最佳模型

| 主要指標 | 最佳方法 | Threshold | Test 結果 |
|---|---|---:|---:|
| Precision | Method 6：CV + Self-training | 0.7 | 0.5862 |
| Recall | Method 7：CV + GAN + Self-training | 0.7 | 0.2559 |
| mAP50 | Method 6：CV + Self-training | 0.5 | 0.1975 |
| mAP50-95 | Method 7：CV + GAN + Self-training | 0.6 | 0.1007 |

## 實驗結論

1. Baseline 在日間訓練資料上的表現較好，但在夜間 Test set 上的 Recall 與 mAP 明顯下降。
2. 單獨使用 CV preprocessing 可以改善部分夜間泛化能力，其中 saturation = 0.00 的結果相對較好。
3. GAN preprocessing 能提升模型在夜間 Test set 上的 mAP50 與 mAP50-95。
4. CV preprocessing 與 GAN 結合後，能進一步提升 Recall 與 mAP。
5. Self-training 的效果高度依賴 confidence threshold。
6. threshold 過高時，pseudo-label 數量可能不足，導致模型泛化能力下降。
7. Method 7 在 threshold = 0.6 時取得最高的 Test mAP50-95。
8. 最佳模型的 Test mAP50-95 為 0.1007，相較於 Baseline 的 0.0483 提升約 108.7%。
9. 最佳模型並非 Precision 最高的模型，因此模型選擇應根據實際應用需求決定。

## 研究限制

1. Nighttime dataset 只包含 camera 3 與 camera 4。
2. 訓練資料也限制為 camera 3 與 camera 4，因此結果不一定能代表所有 camera。
3. 所有實驗主要使用單一 YOLOv8s 模型。
4. 實驗結果可能受到訓練 epoch、batch size、image size 與 random seed 影響。
5. 目前部分實驗缺乏多次重複實驗，因此尚未計算平均值與標準差。
6. GAN 影像轉換品質會影響後續物件偵測模型的訓練結果。
7. Self-training 使用 pseudo-label，錯誤 pseudo-label 可能被模型進一步學習。
8. 目前主要比較整體指標，尚未針對不同物件類別進行詳細分析。
9. Precision、Recall 與 mAP 之間存在 trade-off，單一最佳 threshold 不一定適用於所有應用情境。

## Citation

### FishEye8K

本研究使用 FishEye8K dataset：

```bibtex
@InProceedings{Gochoo_2023_CVPR,
  author    = {Gochoo, Munkhjargal and Otgonbold, Munkh-Erdene and
               Ganbold, Erkhembayar and Hsieh, Jun-Wei and
               Chang, Ming-Ching and Chen, Ping-Yang and Dorj, Byambaa and
               Al Jassmi, Hamad and Batnasan, Ganzorig and Alnajjar, Fady and
               Abduljabbar, Mohammed and Lin, Fang-Pang},
  title     = {FishEye8K: A Benchmark and Dataset for Fisheye Camera Object Detection},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  month     = {June},
  year      = {2023},
  pages     = {5304--5312}
}
```

- Dataset: https://huggingface.co/datasets/Voxel51/fisheye8k
- Repository: https://github.com/MoyoG/FishEye8K
- License: CC BY-NC-SA 4.0

### img2img-turbo

本研究使用 img2img-turbo 進行影像日夜風格轉換：

```bibtex
@article{parmar2024one,
  title   = {One-Step Image Translation with Text-to-Image Models},
  author  = {Parmar, Gaurav and Park, Taesung and Narasimhan, Srinivasa and Zhu, Jun-Yan},
  journal = {arXiv preprint arXiv:2403.12036},
  year    = {2024}
}
```

- Repository: https://github.com/GaParmar/img2img-turbo
- Repository license: MIT License
- Paper: https://arxiv.org/abs/2403.12036

### Ultralytics YOLO

本研究使用 Ultralytics YOLOv8s 進行物件偵測：

- Repository: https://github.com/ultralytics/ultralytics
- Citation metadata: https://github.com/ultralytics/ultralytics/blob/main/CITATION.cff
- License: AGPL-3.0 License

