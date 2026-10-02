# Deteksi Keberadaan Tanda Tangan pada Ijazah (Thresholding + Morfologi)

Program mendeteksi apakah area tanda tangan **Dekan** pada citra ijazah terisi
(`SIGNATURE PRESENT`) atau kosong (`SIGNATURE ABSENT`) memakai pengolahan citra klasik (OpenCV),
tanpa machine learning.

> Catatan: pada tugas tertulis "kepala sekolah". Dokumen ini ijazah universitas, jadi area yang
> dipakai adalah tanda tangan **Dekan** (kanan atas blok tanda tangan).

---

## 1. Cara menjalankan

```bash
# 1. (opsional) buat virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. pasang dependensi
pip install -r requirements.txt   # opencv-python, numpy, matplotlib

# 3. jalankan (9 citra sudah ada di folder images/)
python signature_detection.py --input_dir ./images --output_dir ./output
```

Opsi:

| Argumen | Default | Keterangan |
|---|---|---|
| `--input_dir` | `./images` | Folder citra `.jpg`/`.png` |
| `--output_dir` | `./output` | Folder hasil |
| `--method` | `otsu` | Metode untuk keputusan akhir: `global`, `otsu`, `adaptive` |

Struktur folder:

```
signature_detection/
├── signature_detection.py   # program utama
├── requirements.txt
├── README.md
├── images/                  # 9 citra uji
└── output/
    ├── results.csv          # semua fitur + keputusan per citra/ROI/metode
    ├── accuracy.txt         # akurasi tiap metode
    └── figures/             # crop ROI, visualisasi pipeline, threshold sweep
```

Jika dipakai pada citra lain: sesuaikan `ROIS` (posisi crop, dalam fraksi) dan rotasi di
`load_upright()` bila orientasi scan berbeda.

---

## 2. Alur pipeline

1. **Rotasi + crop ROI.** Scan tersimpan miring 90°, diputar tegak, lalu area tanda tangan Dekan
   di-crop dan dinormalkan ke lebar 800 px supaya parameter konsisten.
2. **Grayscale** + Gaussian blur 3×3.
3. **Thresholding, tiga metode dibandingkan:**
   - *Global*: T tetap = 110
   - *Otsu*: T dihitung otomatis dari histogram (+ validasi kontras, lihat bagian 4)
   - *Adaptive Gaussian*: block 51, C = 12
4. **Morfologi:** *opening* elips 3×3 (buang bintik noise) → *closing* elips 9×9 (sambung goresan putus).
5. **Karakteristik area:** jumlah piksel foreground, rasio foreground, jumlah komponen,
   luas serta lebar/tinggi bounding box komponen terbesar (connected component).
6. **Aturan keputusan** (semua harus terpenuhi → `PRESENT`):

   | Kondisi | Nilai |
   |---|---|
   | rasio foreground | 1% – 30% dari ROI |
   | luas komponen terbesar | ≥ 1500 px |
   | lebar bbox komponen terbesar | ≥ 35% lebar ROI |
   | tinggi bbox komponen terbesar | ≥ 30% tinggi ROI |

   Tanda tangan berupa goresan panjang yang saling menyambung, sedangkan teks cetak terpecah
   menjadi huruf/kata kecil. Karena itu bentuk komponen dipakai, bukan hanya jumlah piksel.

---

## 3. Skenario pengujian dan hasil

**Sampel positif (9):** ROI tanda tangan Dekan pada 9 versi citra (high quality, low contrast,
blur, noise, low resolution, faded, warm tint, JPEG artifacts, kombinasi).

**Sampel negatif (18):** pada tiap citra, dua ROI tanpa tanda tangan:
- `blank`: kertas kosong (hanya tekstur pola pengaman)
- `typed_text`: baris teks cetak "Dr. Ir. Petrus Mursanto, M.Sc." (negatif sulit: ada piksel gelap tetapi bukan tanda tangan)

**Akurasi (27 pengujian):**

| Metode | Benar | Akurasi |
|---|---|---|
| Global (T = 110) | 20 / 27 | 74,1% |
| **Otsu** | 27 / 27 | **100%** |
| Adaptive Gaussian | 27 / 27 | **100%** |

**ROI tanda tangan, jumlah piksel foreground setelah morfologi:**

| Citra | Global | Otsu | Adaptive |
|---|---|---|---|
| 01_HighQuality_Enhanced | 16457 / PRESENT | 24478 / PRESENT | 26454 / PRESENT |
| 02_LowContrast | 0 / **ABSENT** | 24492 / PRESENT | 24979 / PRESENT |
| 03_Blurred | 4557 / **ABSENT** | 31699 / PRESENT | 30391 / PRESENT |
| 04_HighNoise | 12912 / **ABSENT** | 24058 / PRESENT | 25873 / PRESENT |
| 05_LowResolution_Upsampled | 10327 / **ABSENT** | 26338 / PRESENT | 27770 / PRESENT |
| 06_Faded_Underexposed | 18516 / PRESENT | 24607 / PRESENT | 23943 / PRESENT |
| 07_ColorShift_WarmTint | 13644 / **ABSENT** | 24567 / PRESENT | 26259 / PRESENT |
| 08_JPEGCompression_Artifacts | 12567 / **ABSENT** | 24541 / PRESENT | 26317 / PRESENT |
| 09_CombinedDegradation | 14574 / **ABSENT** | 25709 / PRESENT | 25984 / PRESENT |

Pada ROI `blank`, foreground = 0 piksel di semua metode. Pada ROI `typed_text`, foreground
Otsu 10.730–16.150 piksel (setara atau lebih dari beberapa tanda tangan), tetapi tetap `ABSENT`
karena komponen terbesarnya hanya selebar satu kata. Jadi jumlah piksel saja tidak cukup sebagai aturan.

**Perbandingan metode:**
- **Global** hanya benar pada citra 01 dan 06. Pada citra 02 (low contrast) goresan tertinggi
  nilai gelapnya hanya ±103, sehingga tidak ada piksel lolos T = 110 (foreground = 0). Pada citra blur, noise,
  low-res, warm tint, JPEG, dan kombinasi, goresan tipis menjadi lebih terang daripada T sehingga
  terputus-putus (lebar komponen terbesar hanya ±30% ROI, di bawah syarat 35%).
- **Otsu** menyesuaikan T pada tiap citra (contoh citra 09: T ≈ 145, sedangkan kertas ±190),
  sehingga stabil di semua degradasi.
- **Adaptive** juga 100% dan hasil goresannya sedikit lebih tebal/utuh; kelemahannya adalah
  sensitif terhadap tekstur lokal sehingga bergantung pada opening untuk menekan noise.

Visualisasi: `output/figures/*_pipeline.png` (crop → grayscale → threshold → morfologi → fitur → keputusan)
dan `output/figures/*_threshold_sweep.png` (pengaruh nilai T).

---

## 4. Analisis

### Mengapa thresholding diperlukan sebelum analisis keberadaan tanda tangan?

Citra grayscale hanya berisi intensitas 0–255; belum ada pemisahan objek vs latar. Untuk
menjawab "ada tanda tangan atau tidak" kita perlu mengukur *seberapa banyak piksel tinta* dan
*bagaimana bentuknya*, dan itu hanya bisa dilakukan pada citra biner:

1. **Segmentasi:** memisahkan goresan tinta (foreground) dari kertas (background), sehingga luas,
   rasio, dan connected component bisa dihitung.
2. **Menghilangkan variasi latar:** kertas ijazah berwarna dan bertekstur. Setelah biner,
   warna/tekstur halus tidak lagi mempengaruhi analisis.
3. **Syarat operasi lanjutan:** morfologi (opening/closing) dan connected component bekerja pada citra biner.
4. **Fitur yang sederhana dan dapat dijelaskan:** aturan seperti "foreground ≥ 1% dan komponen
   terbesar ≥ 35% lebar ROI" hanya mungkin dibuat setelah ada mask foreground.

Perlu dicatat, pada kertas kosong Otsu tetap akan membelah histogram (karena memang selalu membagi
dua kelas). Karena itu program menambah validasi: jika rata-rata latar dan foreground berbeda < 45 level
abu-abu, hasil dianggap bukan tinta dan mask dikosongkan. Tanpa ini, tekstur kertas akan terbaca
sebagai tanda tangan.

### Apa masalah jika threshold terlalu tinggi atau terlalu rendah?

Dengan `THRESH_BINARY_INV` piksel dengan nilai < T menjadi foreground (lihat `*_threshold_sweep.png`,
citra 01, kertas ±247):

| | Threshold terlalu **rendah** (mis. T = 30) | Threshold terlalu **tinggi** (mis. T = 250) |
|---|---|---|
| Hasil | Hanya inti goresan tergelap lolos; goresan tipis hilang | Tekstur kertas ikut menjadi foreground |
| Angka (citra 01) | foreground 0,5% dan goresan terpecah | foreground 91,3% (hampir seluruh ROI) |
| Dampak pada aturan | komponen terbesar terlalu kecil/sempit → **false negative** (tanda tangan tipis/pudar dianggap absen) | rasio foreground melewati batas dan komponen mengisi seluruh ROI → **false positive** atau pengukuran bentuk tidak bermakna |
| Dampak lain | Pada citra low contrast (02) nilai T = 110 menghasilkan 0 piksel | Noise, bercak, dan tepi kertas terbaca sebagai tinta |

Nilai T yang baik berada di antara dua puncak histogram (tinta vs kertas), dan posisinya bergeser
mengikuti pencahayaan, kontras, dan degradasi. Itu alasan metode otomatis (Otsu/adaptive) lebih
andal daripada T tetap: pada percobaan ini T tetap = 110 gagal pada 7 dari 9 citra bertanda tangan.

---

## 5. Keterbatasan (perlu diketahui)

- **Sampel positif bukan 9 tanda tangan berbeda.** Kesembilan citra adalah satu ijazah yang sama dengan
  degradasi berbeda, jadi menguji ketahanan terhadap degradasi, bukan variasi gaya tanda tangan.
- **Sampel negatif dibuat dari area lain pada ijazah yang sama** (kertas kosong dan teks cetak), bukan
  ijazah lain yang benar-benar tidak ditandatangani. Jenis negatif lain (coretan, stempel, cap
  tinta, noda) belum diuji.
- **Parameter aturan** (1%, 1500 px, 35%, 30%) saya pilih setelah melihat citra ini, sehingga akurasi
  100% bersifat optimistis. Pada dokumen lain perlu divalidasi ulang.
- **ROI berupa koordinat tetap (fraksi)** dan mengasumsikan tata letak yang sama. Bila posisi
  tanda tangan bergeser atau scan dipotong berbeda, ROI harus disesuaikan atau dideteksi otomatis.
- Teks label "Dekan" dan inisial kecil di tepi ROI dapat ikut masuk, tetapi tidak mengubah keputusan
  karena aturan memakai komponen terbesar.
