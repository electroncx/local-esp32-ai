# local-esp32-ai: details

The [README](../README.md) is the short version. This is how it works and how it was tested.

## Where it comes from

Built by following the [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
curriculum and shrinking every piece down to microcontroller size:

| Curriculum lesson | Used here |
|---|---|
| [07-07 GPT causal language modeling](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/07-gpt-causal-language-modeling) | `train/model.py`, a 6-layer decoder-only transformer |
| [07-15 Attention variants](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/15-attention-variants) | grouped-query attention: 5 query heads share 1 K/V head |
| [10-01 Tokenizers](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/10-llms-from-scratch/01-tokenizers) | character tokenizer, 97 tokens, no vocab file needed |
| [10-06 Instruction tuning (SFT)](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/10-llms-from-scratch/06-instruction-tuning-sft) | loss on answer tokens only, so the model learns to answer and stop |
| [10-11 Quantization](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/10-llms-from-scratch/11-quantization) | 2-bit weights with fp16 scales, quantization-aware training |
| [07-12 KV cache](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/12-kv-cache-flash-attention) / [10-12 Inference optimization](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/10-llms-from-scratch/12-inference-optimization) | `tinyai.c`: integer matmuls, int8 KV cache, both cores |

## Survival knowledge

288 facts. 198 are in `data/facts_survival.tsv`, across the 15 topics below. 90 more, in `data/facts_survival_more.tsv`, cover child and infant CPR, fire extinguishers and kitchen fires, CB and marine radio, telling time by the sun, foraging, nuclear and building-collapse safety (NFPA, FCC, US Coast Guard, AHA, DHS). Each answer follows the source named in its section:

| Topic | Facts | Examples | Source |
|---|---|---|---|
| Priorities, being lost, kits | 18 | rule of threes, stay put, what to pack | US Army FM 21-76, Ready.gov |
| Water | 25 | boiling, bleach and iodine doses, SODIS, finding water, seawater, ORS recipe | CDC, EPA, WHO, FM 21-76 |
| Fire | 14 | tinder, fire without matches, wet wood, putting it out | US Forest Service, FM 21-76 |
| Shelter and warmth | 10 | debris hut, lean-to, snow cave, layers, cotton | FM 21-76, NWS |
| Signaling and rescue | 9 | 3 of anything, SOS, signal mirror, ground-to-air V and X | FM 21-76, ICAO, FCC |
| Navigation | 9 | North Star, Southern Cross, shadow stick, watch method, moss myth | FM 21-76 |
| Food | 10 | insects, mushrooms, berries, edibility test, milky sap | FM 21-76 |
| Cold | 20 | hypothermia, frostbite, falling through ice, ice thickness, 1-10-1 | CDC, MN DNR, US Coast Guard |
| Heat | 7 | heat exhaustion vs heat stroke, desert travel | CDC |
| Wildlife | 16 | black vs grizzly bears, mountain lions, moose, snakebite, ticks | National Park Service, CDC, WHO |
| Weather and disasters | 29 | lightning, tornado, flood, tsunami, wildfire, avalanche, power outage, generators | NWS, Ready.gov, FoodSafety.gov |
| Water hazards | 6 | rip currents, sinking car, river crossing, capsized boat | NOAA, US Coast Guard |
| First aid | 15 | bleeding, tourniquet, splints, shock, altitude sickness | Stop the Bleed, Red Cross, NHS, CDC |
| Knots | 6 | bowline, sheet bend, taut-line, clove, trucker's hitch | |
| Gear | 3 | survival knife, paracord | |

**How the answers were chosen.** Only guidance published by the agencies above, in their words where possible, cut to 48 characters. Where sources disagree, the more cautious answer wins. For example, a debris hut's leaf pile is given as 3+ feet (1 m), not the thinner 2 feet some guides say. Regional advice that doesn't generalize was left out: snakebite pressure bandages are Australian practice, and jellyfish advice depends on the species.

**Survival math is computed, not memorized:**

| You type | It answers | Rule |
|---|---|---|
| `how much water for 4 people for 3 days` | `12 gallons (45 L) for 3 days.` | Ready.gov: 1 gallon per person per day |
| `how much water should i keep for a family of 4` | `4 gallons (15 L) a day.` | |
| `thunder 10 seconds after lightning` | `About 2.1 miles (3.4 km). Go indoors.` | sound travels 343 m/s; NWS: if you hear thunder, go in |
| `how much bleach for 4 gallons` | `32 drops of 6% bleach; wait 30 min.` | CDC/EPA: 8 drops per gallon, 2 per liter |

**How the gate handles survival questions.** People in trouble describe their situation ("how to survive in *extreme* heat", "whats the *fastest* way to purify water"). Survival facts are *lenient*: the question may add one word of context that no phrasing contains. That word is never allowed to be:
- a word some other survival answer is about: "how long is **freezer** food good when the power is out" must not get the fridge answer
- the method, after "with" or "using": "purify water **with a lifestraw**" is not "boil it"
- who it's for, or a number or negation: "my **friend** fell through the ice" needs rescue advice, not self-rescue, and "**baby**", "**dog**" and "**5** gallons" all change the answer

Each of those rules came from a wrong answer found in testing ([below](#survival-questions)).

## How a question is answered

```
"Whats the best way to purify water??"
   │ normalize         lowercase, strip junk            "whats the best way to purify water"
   │ compute           math, money, units, dates,        not computable
   │                   water, bleach, lightning
   │ knowledge gate    indexed match, 146,757 phrasings  fact #3,307  ("what's the best way" = "how")
   │                   no match -> "I don't know."
   │ errata            fact stored as text? answer it   (or refuse: model wrong, no room)
   │ model (2-bit GPT) fact's shortest phrasing -> answer "how to clean water"
   ▼                   greedy, stops at EOS, 48 chars max
"Boil it for 1 minute (3 above 5,000 ft)."
```

## How it stays direct

Directness is enforced at every layer, not just requested in a prompt:

1. **Data.** Every training answer in `data/facts*.tsv` is the answer only: no preamble, no restated question.
2. **Loss.** The loss covers answer characters only. The model never learns to write anything before the answer.
3. **Stop token.** Each answer ends in EOS, and generation stops there.
4. **Greedy decoding.** It always takes the most likely token: no sampling, no temperature, no rambling.
5. **Hard cap.** An answer is at most 48 characters, enforced in C.
6. **Knowledge gate** (`gate.c`). A tiny model can't tell what it doesn't know. Ask for a capital it never saw and it will invent one. The gate checks the question against every phrasing the model was trained on. It tolerates typos, plurals, split words ("humming bird") and filler, and it requires the question type to agree: a "how many" question never gets a "what" answer. If two different facts match equally well, it refuses rather than guess. With no match, the answer is `I don't know.`
7. **Computed answers** (`calc.c`, `convert.c`, `everyday.c`). Anything that can be calculated is, rather than memorized, and number words work too ("twelve times seven", "half of 30", "a dozen"):
   - math: `12*7`, `15% of 80`, `50 plus 10%` (= 55, like any calculator), `square root of 144`, `average of 3, 5 and 10`
   - money: `15% tip on 42.50` → `Tip 6.38, total 48.88.`, `8% tax on 50`, `20% off 80`, `what percent is 12 of 48`, `percent change from 50 to 75`, `split 90 between 4 people`
   - 63 units: `10 km in miles`, `how many pounds is 70 kg`, `350 f in c`, `32 psi in bar`
   - times: `3pm in 24 hour time`, `how long from 9am to 5:30pm`, `3pm eastern in pacific`, `10:30 utc in jst`
   - dates: `what day was july 20 1969`, `days between march 3 and june 10`, `30 days after march 3 2025`, `is 2100 a leap year`
   - survival: water for N people and days, bleach for N gallons or liters, distance to lightning
   - health: `bmi 70 kg 175 cm` → `BMI 22.9: healthy weight.`
   - numbers: `255 in hex`, `1994 in roman numerals`, `is 91 prime` → `No, 91 = 7 x 13.`, `10 factorial`
   - chance: `flip a coin`, `roll 2 dice`, `random number between 1 and 10` (seeded from the ESP32's hardware RNG)

   When a computed answer would depend on something it can't know, it says `I don't know.` instead of guessing. For example, "3pm eastern in UTC" depends on daylight saving, and it has no clock.

## Versions

v1 was the first build: a small int8 model that tried to answer from raw text. v2 was a 2M 4-bit model. v3 is the biggest model that fits a 4 MB board. All measured on the C engine:

| | v1 | v2 | **v3 (shipped)** |
|---|---|---|---|
| Parameters | 616K | 2.03M (dim 192) | **5.47M** (dim 320, MLP 1024) |
| Weights | int8 | int4, 4.5 bits each | **int2, 2.5 bits each** |
| Facts answered right | 309 | 24,855 (198 survival) | **33,788** (288 survival) |
| ...straight from the model | | 24,823 | **32,387** |
| Accepted phrasings | 639 | 97,486 | **146,757** |
| Model file (flash) | 720 KB | 2,416 KB | **3,668 KB** |
| Computed questions | arithmetic | + money, units, times, time zones, dates, BMI, bases, primes, dice, survival sums | same |
| Knowledge gate | scans every phrasing | word index, strict for names, lenient for advice | same |
| Matmuls on dual-core chips | 1 core | 2 cores | 2 cores |
| Trained phrasings correct | 97% | 100% | **100%** (146,757) |
| Test questions never trained on, wrong answers | 10 of 68 | 0 of 356 | **0 of 391** |

v2 is still one command away (`python train/train.py`, then `export.py --tier small`), and it answers about 1.5x faster per character on the chip (1.1 MB of weights to stream instead of 1.7 MB).

### Fitting more knowledge in the same chip
- **Retrieval-canonical prompts.** The gate finds the fact, and the model only ever sees that fact's key, never the raw text. Capacity goes to facts instead of typos, and accuracy on the chip matches training. v1 answered "France capital" with "Beijing."; that class of error is gone.
- **Shortest phrasing as the key.** On the chip every prompt character is a full forward pass, so each fact's key is its shortest accepted phrasing ("gold melting point", not "what is the melting point of gold").
- **2-bit weights with quantization-aware training.** 2.5 bits per weight: four levels (−1.5, −0.5, +0.5, +1.5) times one fp16 scale per 32 weights, the scale set to 1.2x the group's mean |w|. The second half of training uses the rounded weights through a straight-through estimator, and `export.py` asserts its rounding matches training bit for bit. At the same flash size, 2-bit fits 2.7x the parameters of v2's int4. That bought 9,000 more facts, not 2.7x the knowledge: a 2-bit weight stores less than a 4-bit one.
- **Flash bandwidth is the real bottleneck.** Weights stream from memory-mapped flash through a 32 KB cache. The 2-bit kernel reads 16 weights per 32-bit load and works on unsigned codes u = 0..3 with a precomputed-sum correction: Σ(u−1.5)·x = (2·Σu·x − 3·Σx) / 2, all in integers until the final scale.
- **Errata, and refusing what doesn't fit.** `export.py` runs the real C engine on every fact's key and collects the ones the model gets wrong (4,901 of 37,288). It stores their correct answers as text, in priority order: every hand-written and survival fact first, always, then Wikidata facts, best known first. It stops when the file reaches its size limit (`--max-bytes`, set so the firmware fits the 4 MB board). The 3,500 left over are lesser-known Wikidata facts, listed in `firmware/data/model.dropped`, and they answer `I don't know.`: a fact the model can't get right is refused, never guessed. Continuing training to fix wrong facts made things worse in v2 (24,850 → 24,811), because every update flips rounding across the whole network.
- **Bit-identical on every chip.** Errata are found on a PC, so the ESP32 must compute exactly the same floats, or a fact that is right on the PC could come out wrong on the board. The first 2-bit build disagreed with the PC on 3 of 350 emulator questions ("biggest planet in the solar system" → "Johnumber, Pope."). The ESP32's FPU fuses multiply-adds, and its libm's `expf`/`tanhf`/`sqrtf` differ from a PC's. Inference now uses only +, −, × and int↔float conversions, which IEEE 754 rounds identically everywhere, with its own exp, reciprocal and inverse square root. Both builds use `-ffp-contract=off`. After the fix: 400/400 identical, including those 3.
- **Grouped-query attention** halves the K/V projections and the KV cache.
- **Both cores.** On the ESP32 and ESP32-S3 each matmul is split across the two cores with FreeRTOS task notifications. Output rows are independent, so answers are bit-identical to single-core.
- **One big app partition.** `firmware/partitions.csv` gives the app 3.9 MB of a 4 MB flash (no OTA). The firmware fills 99.5% of it.

### Knowledge from real sources
- **Wikidata, filtered for reliability.** `data/wikidata_facts.py` queries Wikidata (public domain, CC0) through [QLever](https://qlever.dev) and keeps the best-known items by how many Wikipedias cover them: 4,800 people, 2,500 cities, every country, presidents, mountains, rivers, landmarks, books, films, songs, albums, paintings, companies, universities, wars and battles, compounds, species and programming languages. What it throws away matters as much:
  - dates respect Wikidata's precision (a month-precise date never gets a made-up day), and nothing before 1583
  - birthplaces must be towns or cities, not hospitals; quantities must be in the expected unit
  - song and album credits need a single performer; titles must be distinctive ("Crash" could be a dozen films)
  - only organic chemical formulas, where Wikidata's Hill notation is the familiar form (it writes NaCN as "CNNa")
  - hand-written facts always win over Wikidata ones
- **Names must match in full.** Wikidata facts are *strict*: every word of a phrasing must be in the question, so "where was obama born" can't reach Michelle Obama's birthplace. Barack's birthplace didn't pass the filters, and before this rule that question got Michelle's. Short names are added only where one person clearly dominates: "einstein", "lincoln", "trump" and "obama" (Barack, with 2.5x Michelle's Wikipedia coverage) work, but "roosevelt", "bush", "jackson" and "curie" don't. A name the gate can only partly see is dropped entirely: "Will Smith" is just "smith" to it, since "will" is a filler word.
- **Datasets, not memory.** Every chemical element comes from [mendeleev](https://github.com/lmmentel/mendeleev) (MIT). Currencies come from Unicode CLDR via Babel (BSD), calling codes from phonenumbers (Apache 2.0), US and Canadian abbreviations from pycountry (LGPL), and 20 CODATA physical constants from SciPy (BSD).
- **Everyday knowledge from standard references:** safe cooking temperatures and food storage (USDA / FoodSafety.gov), emergency numbers for 18 countries and the EU, first aid (Red Cross / NHS), health reference numbers (CDC / AHA / FDA / ADA), and home and car settings (US DOE).

## Specs

| | |
|---|---|
| Architecture | GPT: 6 layers, dim 320, 5 query heads / 1 K/V head, MLP 1024, context 96, tied embeddings |
| Parameters | 5,472,640 |
| Knowledge | 33,788 facts answered right of 37,288 stored (the other 3,500 are refused), 146,757 accepted phrasings, plus computed answers |
| Model in flash | 3,668 KB: 2-bit weights and int8 embeddings 1,727 KB, gate index 1,568 KB, fact keys and 1,401 stored answers 373 KB |
| RAM at runtime | about 90 KB: int8 KV cache 77 KB, activations and buffers about 13 KB (235 KB of heap still free on an ESP32) |
| Engine code | pure C99, no dependencies; bit-identical floats on every chip |
| Works on | ESP32, ESP32-S3, ESP32-C3 (4 MB flash) |

## Results

`./test.sh` runs every check below; GitHub CI runs it on every push. Answers are scored on the real C engine (the same code the ESP32 runs), not the Python model:

| Test | Score |
|---|---|
| Every trained phrasing (the questions the gate was built from) | 146,757 / 146,757 (the 3,500 refused facts must answer `I don't know.`, and do) |
| Every fact's key, answered by the model alone (no stored answers) | 32,387 / 37,288 |
| Test questions it never trained on | 372 / 391 right, **0 wrong**, 19 `I don't know.` |
| Test questions that are also trained phrasings | 153 / 154, 0 wrong (1 is a refused fact) |
| Gate index parses like the C gate | 146,756 / 146,756 |
| Answers over 48 chars or starting with filler | 0 |
| ESP32 firmware in the emulator vs the PC build | 350 / 350 identical |

### Test questions

`data/eval.tsv` and `data/eval_more.tsv` hold 545 questions, each typed as a person would (missing apostrophes, no capitals, extra words, typos), each with one exact expected answer. The C engine must produce that answer character for character: a close answer counts as wrong. 154 of them turned out to match a trained phrasing exactly; they are scored separately, and the other 391 are the "never trained on" score.

| Group | Examples | Right | Wrong | Refused |
|---|---|---|---|---|
| Rephrased general facts | "France capital", "hummingbird weight", look-alikes: Iceland vs Ireland, Niger vs Nigeria | 83 / 83 | 0 | 0 |
| Calculator and conversions | "(3+4)*5", "1/0", "10 km in miles", "1994 in roman numerals" | 22 / 22 | 0 | 0 |
| Must refuse | "how many moons does jupiter have", "how far is pluto" | 19 / 19 | 0 | 0 |
| Everyday, batch 1 | "whats the safe temp for chicken", "15% tip on 42.50", 10 must-refuse | 90 / 90 | 0 | 0 |
| Everyday, batch 2 | 85 more, written after batch 1 was tuned | 85 / 85 | 0 | 0 |
| Wikidata knowledge | "einstein birthday", "how high is kilimanjaro", "who was roosevelt" (must refuse: ambiguous) | 38 / 40 | 0 | 2 (facts the model gets wrong, refused) |
| Survival, batches 1-3 | "can i eat snow if im thirsty", "a moose is charging at me" | 167 / 170 | 0 | 3 |
| Survival, batch 4 | "how hard do i push for cpr on a toddler", "are toads edible" | 21 / 36 | 0 | 15 |
| **Total** | | **525 / 545** | **0** | **20** |

"Refused" means `I don't know.`: safe, but a miss. "Wrong" means a different answer, the failure this project is built to prevent.

#### Survival questions

The survival tests were written in four batches, each *before* looking at how the gate handled it, and each first score is recorded here as it was:

| Batch | First run, before any change | What was fixed afterwards |
|---|---|---|
| 1: 69 questions | 34 right, **0 wrong**, 33 refused | synonyms (frostbitten, forest, elevation, "no matches"), more phrasings |
| 2: 55 questions | 22 right, **0 wrong**, 33 refused | lenient advice facts; "what's the best way to" = "how" |
| 3: 47 questions | 23 right, **5 wrong**, 19 refused | the leniency rules above: freezer vs fridge, "with a flint", "my friend", "how can I **tell** if" |
| 4: 35 questions (the 90 newer facts) | 20 right, **0 wrong**, 15 refused | nothing: left as a measure of unseen wording |

The honest takeaway: **on survival questions worded in a way it hasn't seen, expect it to answer about half and decline the rest.** Batch 3 is the warning: the first version of the leniency rule gave 5 wrong answers in 47 questions. Today all 545 test questions give either the right answer or `I don't know.`, but a new batch could find a new way to be wrong. Keep questions short and plain ("signs of hypothermia", "how to purify water") for the best results.

**What "never trained on" means, precisely.** The model never saw any of these questions. The knowledge gate was adjusted while looking at the results of each batch, so after its fixes each batch is a regression test, not an unbiased measure. The first-run scores above are the unbiased ones.

**How it can still be wrong:**
- **Unusual phrasing** can match a similar but different fact. The gate checks words, question type and coverage, but it matches words, not meaning.
- **Facts can be outdated or simplified.** Populations, leaders and guidelines change, and a 48-character answer leaves out detail. Some answers are US-specific (gallons, 911) and say so where it matters.
- **Survival advice is general.** Local conditions, species and injuries vary. A short answer is a reminder, not training.
- **A lesser-known Wikidata fact may be refused even though it's stored.** The 3,500 the model gets wrong answer `I don't know.` ("population of chicago", "how long is the mississippi" among them).
- **Speed on a physical board hasn't been measured,** only correctness in the emulator.

### Verified on ESP32 firmware

- **Builds** with PlatformIO for all three chips with no warnings:

  | Board | Flash used (of the 3.9 MB app partition) | Static RAM |
  |---|---|---|
  | ESP32 | 4,011 KB (99.5%) | 34.1 KB |
  | ESP32-S3 | 4,008 KB (99.4%) | 31.2 KB |
  | ESP32-C3 | 3,993 KB (99.0%) | 26.6 KB |

  `esp32dev` flashes in QIO mode at 80 MHz (answers stream weights from flash). `esp32dev-dio` is the same firmware in DIO mode, for boards that won't boot in QIO and for the emulator.
- **Runs** in Espressif's ESP32 emulator (QEMU) as the real flash image, on both cores: it boots with 235 KB of heap free and gives exactly the PC build's answer to 350 questions (150 test questions and 200 trained facts the model answers itself), so the float math is bit-identical. The ESP32-S3 and ESP32-C3 builds compile but can't be run in this emulator; they use the same engine code and the same IEEE float operations.
- **Not yet timed on physical hardware.** Emulator timings aren't real. Each answer streams the 1.7 MB of 2-bit weights from flash once per character, so a rough, unmeasured estimate is several seconds per model answer; computed answers, stored answers and refusals take milliseconds. To measure on a board, set `SHOW_TIMING 1` in `firmware/src/main.cpp`.

Re-run the emulator check yourself (needs [Espressif's QEMU](https://github.com/espressif/qemu/releases)):

```sh
cd firmware && pio run -e esp32dev-dio
echo "how do i purify water" | python qemu_test.py --qemu path/to/qemu-system-xtensa
./test.sh --qemu path/to/qemu-system-xtensa      # everything, as CI runs it
```

## Layout

```
data/facts.tsv              hand-written everyday facts
data/facts_survival.tsv     hand-written survival facts, with sources (edit these)
data/facts_survival_more.tsv 90 more survival facts (the shipped max4mb tier)
data/more_facts.py          dataset facts -> data/facts_generated.tsv
data/wikidata_facts.py      Wikidata facts -> data/facts_wikidata.tsv (small tier),
                            data/facts_wikidata_max4mb.tsv (--tier max4mb, shipped)
data/eval.tsv, eval_more.tsv test questions: paraphrases, everyday, survival, must-refuse
train/common.py             tokenizer, normalization, fact tiers (mirrored in C)
train/model.py              the transformer, int4/int2/ternary fake-quantization for QAT
train/train.py              training (resumable: --resume, which checks the run matches)
train/export.py             quantization + export, errata, --max-bytes
train/gate_index.py         builds the gate's word index (mirrors gate.c)
train/check_routing.py      fast check: every phrasing reaches its fact
train/evaluate.py           scores the real C engine
firmware/data/model.bin     the shipped model (+ model.tier, its fact set; model.dropped, facts it refuses)
firmware/lib/tinyai/        inference engine, gate, calculator, units, everyday and survival math (C99)
firmware/src/main.cpp       serial chat for the ESP32
firmware/partitions.csv     one 3.9 MB app partition for the model
firmware/qemu_test.py       runs the built firmware in the ESP32 emulator
host/                       PC build of the same engine
test.sh                     every check in one command (CI: .github/workflows/ci.yml)
```

## Limits

- It knows what's in `data/facts*.tsv` and nothing else. Anything else gets `I don't know.`, which is the point.
- No clock, GPS or internet: "what time is it", "where am I", "weather" and "news" can't be answered.
- First-aid, health and survival answers are short reference facts, not medical advice or training.
- The gate matches words, not meaning. Heavily reworded questions can be refused even when the fact is known. Add those phrasings to the fact's line.
- 5.5M parameters can store facts. They can't reason. Multi-step questions are out of scope.
- Numbers from datasets are as good as the datasets: Wikidata as queried in September 2026, element data from mendeleev, currencies from CLDR as of Babel 2.18.
