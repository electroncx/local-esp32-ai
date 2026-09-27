# How to use it

## 1. Set up

**First time (once, with a computer):**

1. Download this project: on GitHub click **Code → Download ZIP** and unzip it. No GitHub account needed.
2. Install [VS Code](https://code.visualstudio.com) and its **PlatformIO** extension.
3. Plug the ESP32 into the computer with a USB cable.
4. Open a terminal in the project's `firmware` folder and run `pio run -t upload`. This copies the AI onto the ESP32 (about a minute).

**Every time after that:**

1. Plug the ESP32 into any computer (or a phone with a USB serial app). No internet, no GitHub, no uploading again: the AI lives on the ESP32.
2. Open a serial terminal at **115200 baud**: `pio device monitor` in the `firmware` folder. The terminal is just a keyboard and screen for the ESP32.
3. Wait for `tinyai ready`, type a question, press Enter.
4. **Keep it short.** `hypothermia` works; `i think my friend has hypothermia what do i do` doesn't.
5. `I don't know.` means no matching fact: try fewer words, or it doesn't know.

## 2. Ask

Type a question and press Enter. It answers in one short line.

```
you: how to purify water
esp: Boil it for 1 minute (3 above 5,000 ft).
```

No capitals, punctuation or perfect spelling needed.

## 3. Ask short

This matters most. **Name the topic in a few words:**

| Works | Doesn't |
|---|---|
| `hypothermia` | `i think my friend has hypothermia what do i do` |
| `grizzly bear attack` | `a huge bear is running at me in the woods` |
| `how to stop bleeding` | `my leg got cut really bad on a rock and won't stop` |

It matches words, not meaning, so every extra word is another chance to miss. When a long question gets `I don't know.`, try again with just the key words.

## What to ask

**Survival**

```
purify water                         Boil it for 1 minute (3 above 5,000 ft).
can i eat snow                       Melt it first; eating it chills you.
how to start a fire without matches  Ferro rod, lens, bow drill, or battery and wool.
what do i do if i get lost           Stop and stay put, keep warm, signal for help.
sos morse code                       ... --- ... (3 short, 3 long, 3 short).
tornado safety                       Basement or small interior room, lowest floor.
```

Topics: water, fire, shelter, signaling, navigation, wild food, cold, heat, animals, weather and disasters, rip currents and ice, first aid, knots.

**Survival math** (put the numbers in)

```
how much water for 2 people for 5 days   10 gallons (38 L) for 5 days.
how much bleach for 2 gallons            16 drops of 6% bleach; wait 30 min.
lightning 6 seconds                      About 1.3 miles (2.1 km). Go indoors.
```

**Everything else**

```
capital of japan        Tokyo.
who wrote hamlet        William Shakespeare.
10 km in miles          6.21371 miles
20% tip on 35           Tip 7, total 42.
```

## What the answers mean

- **A short answer** is the whole answer. It never adds anything before or after it.
- **`I don't know.`** means it has no fact that matches. It's not broken: rephrase shorter, or it just doesn't know. It will never make something up to fill the gap.
- **`I have no clock.`** and similar: it has no clock, GPS or internet, so it can't tell the time, weather or news.

## Be careful

- **It might make a mistake.** Double-check anything that matters.
- **In an emergency, call for help first** (911, 112, 999 or your local number) if you have any signal. Use this when you can't.
- Survival and first-aid answers are short reminders, not training or medical advice.

## PC-only extras

```sh
./host/tinyai firmware/data/model.bin --why    # also shows which fact it matched
./host/tinyai firmware/data/model.bin --time   # shows how long each answer took
```

To see timing on the ESP32, set `SHOW_TIMING 1` in `firmware/src/main.cpp` and re-flash.
