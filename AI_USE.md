## Role A — Value-based dispatcher (Sezen Balkan)
Bu kısım baştan sona benim işimdi ve kararların hepsini ben verdim. Ajan başta hiç şarj etmiyordu,
sonra "passive collapse" denen şeye düştü — bunu teşhis ettim. En kritik anı, time özelliğinin
0–500 aralığında ham gittiğini ve ağı domine ettiğini fark edip onu normalize ettiğimde yaşadım; tek
hamlede en büyük sıçramayı orada aldık. Double DQN'e ve n-step'e geçmeye ben karar verdim, 3M ve 6M
step'e geçip incelemeyi kararı verdim ve diverjansı yorumladım. En sonunda validasyona bakıp 1M
checkpoint'i submit etmeyi seçtim.

AI'yı burada yardımcı olarak kullandım: tekrar eden kodu yazdırmak, takıldığım yerde debug etmek,
çıkan sayıları çapraz kontrol etmek ve raporu temize çekmek için. Ama hangi deneyi neden koştuğum,
neyin neden bozulduğu ve hangi politikayı seçtiğim bana ait — hepsini tekrar üretebilir ve tek tek
anlatabilirim.
