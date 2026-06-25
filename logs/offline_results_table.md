# Offline RL — sonuç tablosu (ortak iş, Bölüm 20)

Havuz: `offline_pool.npz` = 420,103 transition / 3969 episode (3 üyenin npz'leri birleşik).
Eğitim env'e hiç dokunmadan sabit veride; eval seeds 0,1,2. Kod: `code/offline_rl.py`.

| Yöntem | cost_per_order ↓ | success | final max-Q | yorum |
|---|---|---|---|---|
| BC baseline | 15.22 | 0.56 | — | karışık-kalite veriyi klonlamak zayıf |
| Naive offline DQN | 13.00 | 0.55 | **6263** | Q patlıyor 60→1988→6263 = overestimation |
| **CQL** | **6.61** | 0.72 | 794 | konservatiflik Q'yu sınırlıyor, hem naive hem BC'yi geçer |
| ref: greedy_nearest | 4.57 | — | — | çıta |
| ref: online DQN (1M) | 6.76 | — | — | Role A en iyi |

Ham veri: `logs/offline_qstats.csv` (Q eğrisi), `logs/offline_results.json` (metrikler).

**Sonuç:** (i) naive offline DQN'de overestimation çöküşü ölçülerek gösterildi,
(ii) CQL düzeltti ve hem naive-offline-DQN'i (13.0) hem BC'yi (15.2) geçti → 6.61,
online DQN'e (6.76) denk. Kaynak: Kumar ve ark., Conservative Q-Learning, NeurIPS 2020.

Veri seti: `offline_*.npz` repo'ya konmadı (büyük); `offline_dlogs.zip` ayrı branch'te,
üye npz'leri kendi repolarında. Havuzu yeniden üret: `python pool_offline.py offline_pool.npz offline_dlogs.npz offline_ozan_karhan.npz offline_runa.npz`.
