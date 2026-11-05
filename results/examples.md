# Example outputs

Real pipeline output from this run, copied from [`cases.csv`](cases.csv). For each test category: up to one case that passed every stage and one that did not, with its failure code (meanings in the failure table of [`README.md`](README.md)). Every case is in `cases.csv`.

## plain

**cm_001** (easy, passed)

| | |
|---|---|
| input | aaj mausam bahut accha hai |
| expected Devanagari | आज मौसम बहुत अच्छा है |
| Likho Devanagari | आज मौसम बहुत अच्छा है |
| rewrite (neutral) | आज मौसम बहुत अच्छा है। |
| expected English | The weather is very nice today |
| Likho English | The weather is very good today. |

**cm_002** (easy, failed — **F11**)

| | |
|---|---|
| input | tum kab ghar aaoge |
| expected Devanagari | तुम कब घर आओगे |
| Likho Devanagari | तुम कब घर आओगे |
| rewrite (neutral) | तुम कब घर आओगे? |
| expected English | When will you come home? |
| Likho English | When are you coming home? |

Detail: `F11 chrF++ 42.8`

## loanword

**cm_011** (easy, passed)

| | |
|---|---|
| input | mujhe kal office jana hai |
| expected Devanagari | मुझे कल office जाना है |
| Likho Devanagari | मुझे कल office जाना है |
| rewrite (neutral) | मुझे कल कार्यालय जाना है। |
| expected English | I have to go to the office tomorrow |
| Likho English | I have to go to the office tomorrow. |

**cm_013** (easy, failed — **F10**)

| | |
|---|---|
| input | mera phone charge nahi ho raha |
| expected Devanagari | मेरा phone charge नहीं हो रहा |
| Likho Devanagari | मेरा phone charge नहीं हो रहा |
| rewrite (neutral) | मेरा फोन चार्ज नहीं हो रहा है। |
| expected English | My phone is not charging |
| Likho English | My phone is not charging. |

Detail: `F10 judge meaning < 4`

## named_entity

**cm_023** (easy, passed)

| | |
|---|---|
| input | humne Amazon se naya phone mangwaya |
| expected Devanagari | हमने Amazon से नया phone मंगवाया |
| Likho Devanagari | हमने Amazon से नया phone मंगवाया |
| rewrite (neutral) | हमने Amazon से नया फोन मंगवाया। |
| expected English | We ordered a new phone from Amazon |
| Likho English | We ordered a new phone from Amazon. |

**cm_021** (hard, failed — **F7**)

| | |
|---|---|
| input | Priya aur main kal Delhi ja rahe hain |
| expected Devanagari | Priya और मैं कल Delhi जा रहे हैं |
| Likho Devanagari | प्रिय और मैं कल Delhi जा रहे हैं |
| rewrite (neutral) | कल मैं दिल्ली जा रहा हूँ। |
| expected English | Priya and I are going to Delhi tomorrow |
| Likho English | Tomorrow I am going to Delhi. |

Detail: `F7 Priya: gold Priya, got प्रिय`

## ambiguous

**cm_029** (medium, passed)

| | |
|---|---|
| input | mujhe to pata hi nahi tha |
| expected Devanagari | मुझे तो पता ही नहीं था |
| Likho Devanagari | मुझे तो पता ही नहीं था |
| rewrite (neutral) | मुझे पता ही नहीं था। |
| expected English | I didn't even know |
| Likho English | I didn't know. |

**cm_031** (medium, failed — **F11**)

| | |
|---|---|
| input | ghar me koi nahi hai |
| expected Devanagari | घर में कोई नहीं है |
| Likho Devanagari | घर में कोई नहीं है |
| rewrite (neutral) | घर में कोई नहीं है। |
| expected English | There is no one at home |
| Likho English | No one is at home. |

Detail: `F11 chrF++ 39.6`

## hinglish_verb

**cm_041** (easy, passed)

| | |
|---|---|
| input | kal tak sab adjust ho jayega |
| expected Devanagari | कल तक सब adjust हो जाएगा |
| Likho Devanagari | कल तक सब adjust हो जाएगा |
| rewrite (neutral) | कल तक सब ठीक हो जाएगा। |
| expected English | Everything will be sorted out by tomorrow |
| Likho English | Everything will be fine by tomorrow. |

**cm_037** (easy, failed — **F10**)

| | |
|---|---|
| input | report ko manage karna padega |
| expected Devanagari | report को manage करना पड़ेगा |
| Likho Devanagari | report को manage करना पड़ेगा |
| rewrite (neutral) | — *(rejected: rewrite rejected: script changed from Roman)* |
| expected English | The report will have to be managed |
| Likho English | The report needs to be managed. |

Detail: `F10 rewrite rejected: script changed from Roman`

## long

**cm_043** (hard, failed — **other**)

| | |
|---|---|
| input | kal subah main jaldi uthkar college gaya kyunki pehla lecture aath baje tha, lekin wahan pahunch kar pata chala ki professor ne class cancel kar di hai, toh main library chala gaya aur do ghante tak apne project par kaam kiya, phir dopahar ko doston ke saath canteen mein khana khaya aur shaam ko ghar wapas aa gaya |
| expected Devanagari | कल सुबह मैं जल्दी उठकर college गया क्योंकि पहला lecture आठ बजे था, लेकिन वहां पहुंच कर पता चला कि professor ने class cancel कर दी है, तो मैं library चला गया और दो घंटे तक अपने project पर काम किया, फिर दोपहर को दोस्तों के साथ canteen में खाना खाया और शाम को घर वापस आ गया |
| Likho Devanagari | कल सुबह मैं जल्दी उठकर college गया क्यूंकि पहला lecture आठ बजे था, लेकिन वाहन पहुंच कर पता चला की professor ने class cancel कर दी है, तोह मैं library चला गया और दो घंटे तक अपने project पर काम किया, फिर दोपहर को दोस्तों के साथ canteen में खाना खाया और शाम को घर वापस आ गया |
| rewrite (neutral) | कल सुबह मैं कॉलेज जल्दी उठकर गया क्योंकि पहला lecture आठ बजे था, लेकिन मुझे पता चला की प्रोफेसर ने क्लास को रद्द कर दिया है, इसलिए मैं पुस्तकालय गया और दो घंटे तक अपने प्रोजेक्ट पर काम किया, फिर दोपहर को दोस्तों के साथ कैफेटेरिया में खाना खाया और शाम को घर वापस आ गया। |
| expected English | Yesterday morning I got up early and went to college because the first lecture was at eight, but when I got there I found out that the professor had cancelled the class, so I went to the library and worked on my project for two hours, then had lunch with friends in the canteen in the afternoon and came back home in the evening |
| Likho English | Yesterday morning I got up early for college because the first lecture was at eight o'clock, but I found out that the professor had canceled the class, so I went to the library and worked on my project for two hours, then ate lunch with friends in the cafeteria in the afternoon and came back home in the evening. |

Detail: `other kyunki: gold क्योंकि, got क्यूंकि | F8 wahan: gold वहां, got वाहन | F8 ki: gold कि, got की | other toh: gold तो, got तोह`

**cm_044** (hard, failed — **other**)

| | |
|---|---|
| input | dekho yaar, is weekend pe hum sab Goa jaane ka plan bana rahe hain, toh tum bhi apni leave approve karwa lo aur tickets jaldi book kar lena kyunki baad mein flight bahut mehengi ho jaati hai aur hotel bhi nahi milta, baaki details main tumhe raat ko WhatsApp par bhej dunga |
| expected Devanagari | देखो यार, इस weekend पे हम सब Goa जाने का plan बना रहे हैं, तो तुम भी अपनी leave approve करवा लो और tickets जल्दी book कर लेना क्योंकि बाद में flight बहुत महंगी हो जाती है और hotel भी नहीं मिलता, बाकी details मैं तुम्हें रात को WhatsApp पर भेज दूंगा |
| Likho Devanagari | देखो यार, इस weekend पे हम सब Goa जाने का plan बना रहे हैं, तोह तुम भी अपनी leave approve करवा लो और tickets जल्दी book कर लेना क्यूंकि बाद में flight बहुत महंगी हो जाती है और hotel भी नहीं मिलता, बाकी details मैं तुम्हे रात को WhatsApp पर भेज दूंगा |
| rewrite (neutral) | देखो, हम इस सप्ताहांत गोवा जाने की योजना बना रहे हैं, तो क्या तुम अपनी leave approve करवा सकते हो और टिकट जल्दी बुक कर सकते हो, क्योंकि बाद में उड़ानें बहुत महंगी हो जाती हैं और होटल भी नहीं मिलता, बाकी विवरण मैं रात को WhatsApp पर दे दूँगा। |
| expected English | Look, we're all planning a trip to Goa this weekend, so get your leave approved too and book the tickets soon, because flights get very expensive later and hotels aren't available either; I'll send you the rest of the details on WhatsApp tonight |
| Likho English | Look, we are planning to go to Goa this weekend, so can you get your leave approved and book the tickets early, because later the flights become very expensive and the hotel is also not available, I will give the rest of the details on WhatsApp at night. |

Detail: `other toh: gold तो, got तोह | other kyunki: gold क्योंकि, got क्यूंकि | other tumhe: gold तुम्हें, got तुम्हे`

## already_devanagari

**cm_047** (easy, passed)

| | |
|---|---|
| input | मुझे कल जाना है |
| expected Devanagari | मुझे कल जाना है |
| Likho Devanagari | मुझे कल जाना है |
| rewrite (neutral) | मुझे कल जाना है। |
| expected English | I have to go tomorrow |
| Likho English | I have to go tomorrow. |
