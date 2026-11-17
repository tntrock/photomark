# Photomark

替照片加上**看不見、但裁切或截圖後仍能辨識**的浮水印的 Windows 應用程式，以 Rust 撰寫。

概念上類似 SynthID：浮水印不是蓋在角落的 Logo，而是散佈在整張照片每個角落的微弱訊號。
只要留下一小塊畫面，就能驗證「這張圖出自我」並讀回嵌入的識別碼。

Photomark 同時嵌入**兩層**互補的浮水印：

| 層 | 技術 | 擅長 | 弱點 |
| --- | --- | --- | --- |
| 神經網路層 | [Adobe TrustMark](https://github.com/adobe/trustmark)（variant Q，ONNX 推論） | 強力 JPEG 壓縮、大幅縮圖、模糊、調色 | 裁掉超過約 20% 就讀不到 |
| 圖樣層 | 展頻圖磚 + 摺積碼（傳統訊號處理） | 大面積裁切、截圖、裁切後再縮放 | 極小的縮圖、極低品質壓縮 |

檢測時兩層各自判讀，任一層讀到就算偵測到。神經網路層需要另外下載模型（約 65 MB）；
沒有模型時程式仍可運作，只使用圖樣層。

> Photomark 與 SynthID 不相容，也不是 Google 的技術；神經網路層使用的是 Adobe 以 MIT 授權釋出的 TrustMark 模型。
> 它的耐受範圍見下方〈能承受什麼〉與〈限制〉。

## 功能

- **加浮水印**：批次處理照片（JPEG / PNG / WebP / BMP / TIFF），可拖放檔案或資料夾。
  輸出為 `原檔名_wm.jpg/png`，絕不覆寫原檔；保留 ICC 色彩描述檔並套用 EXIF 方向。
- **檢測浮水印**：開啟影像、拖放，或截圖後直接 `Ctrl+V` 貼上。
- **識別碼**：每張照片可嵌入最多 8 bytes 的文字（例如 `ALLEN-26`），檢測時讀回。
- **金鑰**：兩層浮水印都由金鑰決定，沒有金鑰的人無法檢測，也難以針對性移除。
- 另附命令列工具 `photomark-cli`，方便寫腳本。

## 能承受什麼

以 5 張實拍照片（100 萬～1200 萬畫素）、預設強度、兩層皆啟用測試。
嵌入後 PSNR 約 35–41 dB（只用圖樣層時約 37–43 dB），一般觀看下不易察覺：

| 處理 | 結果 | 主要靠哪一層 |
| --- | --- | --- |
| JPEG 重新壓縮（品質 60～85） | 可辨識 | 兩層 |
| JPEG 品質 15 | 5 張中 4 張可辨識 | 兩層 |
| 縮成原圖 12% 再存 JPEG 品質 50 | 縮圖後仍有約 280 px 寬以上時可辨識（5 張中 3 張） | 神經網路層 |
| 高斯模糊（σ=3）、調亮 + 對比 | 可辨識 | 兩層 |
| 大面積裁切（只留 5% 面積） | 可辨識 | 圖樣層 |
| 縮放 0.5×～1.37× | 可辨識 | 兩層 |
| 裁切 + 縮放 + JPEG 同時發生 | 可辨識 | 圖樣層 |
| 截圖（照片被放大/縮小顯示在視窗中再擷取） | 可辨識 | 圖樣層 |
| 旋轉 90°/180°/270° | 可辨識，並回報方向 | 兩層 |
| 鏡射 + 裁切 | 可辨識 | 圖樣層 |
| 未加浮水印的照片、錯誤的金鑰 | 正確回報「未偵測到」 | |

可以用自己的照片重跑這份測試：

```powershell
cargo run --release -p photomark-core --example robustness -- 你的照片.jpg
```

## 限制

- **神經網路層不耐裁切**：TrustMark 是對整張圖編碼，裁掉超過約 20% 後只剩圖樣層可用。
  因此「大幅裁切」加上「極重的壓縮或極小的縮圖」同時發生時仍可能讀不到。
- 裁切後至少要留下約 **2×2 個圖樣週期**（預設約為照片短邊的 1/6 見方）。
  例如 4000×3000 的照片只留 2% 面積時就會失敗。
- 兩層疊加後對畫質的改動比單層大（PSNR 約低 2–3 dB）；在意畫質可調低強度或關閉神經網路層。
- **不支援任意角度旋轉**（例如歪斜 5°）與長寬不等比例的拉伸；只處理 90° 倍數與鏡射。
- 用手機**翻拍螢幕**、重度濾鏡、AI 重繪等會破壞訊號。
- 這是用來**證明出處**的工具，不是防盜拷機制；知道演算法且刻意攻擊的人仍可能把它抹除。
- 金鑰留空時使用內建的公開金鑰，任何人都能檢測；想要只有自己能檢測請自訂金鑰並妥善保存，遺失後無法找回。

## 使用方式

### 下載

到 [Releases](https://github.com/tntrock/photomark/releases/latest) 下載 `photomark-…-windows-x64.zip`，
解壓後執行 `photomark.exe` 即可，不需要安裝。壓縮檔已附模型（`models` 資料夾），神經網路層直接可用。

執行檔沒有數位簽章，第一次開啟時 SmartScreen 可能出現警告，按「其他資訊」→「仍要執行」。

### 安裝神經網路模型

模型檔（`encoder_Q.onnx`、`decoder_Q.onnx`，共約 65 MB）不在原始碼庫內，取得方式擇一：

- 圖形介面上方按「下載模型」，或執行 `photomark-cli fetch-models`。
  會從 Adobe 的模型主機下載到 `%LOCALAPPDATA%\photomark\models`，並以 SHA-256 驗證。
- 使用 Release 的壓縮檔：執行檔旁已附 `models` 資料夾。

程式依序在這些位置尋找模型：環境變數 `PHOTOMARK_MODELS` 指定的資料夾、執行檔旁的 `models`、
`%LOCALAPPDATA%\photomark\models`。

### 圖形介面

執行 `photomark.exe`：

0. 視窗上方會顯示神經網路層的狀態；就緒後可用「啟用」勾選框切換。
1. **加浮水印**分頁：加入照片 → 輸入識別碼（與金鑰，選填）→「開始加上浮水印」。
2. **檢測浮水印**分頁：輸入相同金鑰 → 開啟影像或貼上截圖 → 顯示是否偵測到與識別碼。

強度預設 1.0；調高更耐破壞但平滑區域（天空）可能看得見細微顆粒。

### 命令列

```powershell
photomark-cli embed 輸入.jpg 輸出.jpg --id ALLEN-26 --key "我的金鑰"
photomark-cli detect 可疑圖片.png --key "我的金鑰"
photomark-cli fetch-models            # 下載神經網路模型
```

`detect` 找到浮水印時結束碼為 0，找不到為 1，發生錯誤為 2。
加上 `--no-neural` 可只使用圖樣層。

## 運作原理

### 神經網路層

1. 8 bytes 識別碼以縮短的 BCH(127,92) 碼加上 35 個同位元，成為 99 bits，再與金鑰衍生的遮罩做 XOR。
   金鑰不同（或其他 TrustMark 工具做的浮水印）就無法解出。
2. TrustMark 編碼器在 256×256 的縮圖上算出一張「殘差」，放大回原尺寸後加到照片上
   （長寬比超過 2 時只處理中央的正方形）。
3. 檢測時把影像縮成 256×256、嘗試 4 個旋轉方向交給解碼器，並用軟式解碼修正最不可靠的幾個 bits。
   未加浮水印的影像被誤判的機率約為千萬分之一以下。

### 圖樣層

1. **圖樣**：由金鑰（SHA-256 → PRNG）產生一張 32×32 格的 ±1 圖磚，其中 336 格是固定的同步訊號，
   688 格承載資料。資料為 8 bytes 識別碼 + CRC-16，經 1/2 碼率摺積碼（K=7）編碼後重複 4 次。
2. **嵌入**：圖磚鋪滿整張照片，加在亮度上。強度依局部紋理調整（紋理多的地方藏得多，平滑處藏得少）。
   每格的像素大小依照片尺寸自動選擇（3–16 px）。
3. **檢測**：取出高頻殘差 → 用自相關找出圖樣的重複週期（因此能得知照片被縮放了多少）→
   把整張圖依週期「摺疊」累加成一張圖磚（照片內容互相抵消，浮水印累積）→
   與同步訊號做 FFT 相關，找出位移與 8 種方向 → 讀出資料格，Viterbi 解碼並以 CRC 驗證。

因為圖磚在整張照片上重複，任何一小塊都帶有完整資訊；裁切只是減少了可累加的份數。

## 從原始碼建置

需要 Rust 1.88 以上。

```powershell
cargo build --release
# 產生 target\release\photomark.exe 與 photomark-cli.exe
cargo test
```

- 使用 MSVC 工具鏈（`stable-x86_64-pc-windows-msvc`，需安裝 Visual Studio Build Tools）可直接建置；
  建置時會下載 ONNX Runtime 並靜態連結進執行檔。
- 使用 GNU 工具鏈（`stable-x86_64-pc-windows-gnu`）時，PATH 中需要有 MinGW-w64 的 `dlltool` 與 `as`
  （例如 `winget install BrechtSanders.WinLibs.POSIX.UCRT`）。這種建置在執行時才載入 ONNX Runtime，
  需自行把 [ONNX Runtime 1.22](https://github.com/microsoft/onnxruntime/releases/tag/v1.22.0)
  的 `onnxruntime.dll` 放進模型資料夾。
- 神經網路層的測試（`crates/core/tests/neural.rs`）在找不到模型時會自動略過。
- 發布新版：更新 `Cargo.toml` 的 `version`，推送同名標籤（例如 `v0.1.0`），
  `.github/workflows/release.yml` 會建置、測試並建立 Release。

## 授權

Photomark 以 [GNU General Public License v3.0](LICENSE)（GPL-3.0-only）釋出：
可以自由使用、修改與再發布，但發布修改後的版本時必須同樣以 GPL-3.0 公開原始碼。
內含的第三方元件（TrustMark 模型、ONNX Runtime 等）沿用各自的授權，見 [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt)。

## 致謝

神經網路層使用 Adobe Research 的 [TrustMark](https://github.com/adobe/trustmark)
（Bui et al., *TrustMark: Universal Watermarking for Arbitrary Resolution Images*，MIT 授權）的預訓練模型，
以 [ONNX Runtime](https://onnxruntime.ai/) 執行。

專案結構：

| 路徑 | 內容 |
| --- | --- |
| `crates/core` | 浮水印演算法與影像讀寫（`photomark-core`） |
| `crates/gui` | Windows 圖形介面（`photomark.exe`，egui/eframe） |
| `crates/cli` | 命令列工具（`photomark-cli.exe`） |
