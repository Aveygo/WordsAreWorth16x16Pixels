# Words Are Worth 16x16 Pixels

 [![Aveygo - WordsAreWorth16x16Pixels](https://img.shields.io/static/v1?label=Aveygo&message=WordsAreWorth16x16Pixels&color=black&logo=github)](https://github.com/Aveygo/WordsAreWorth16x16Pixels "Go to GitHub repo")
[![stars - WordsAreWorth16x16Pixels](https://img.shields.io/github/stars/Aveygo/WordsAreWorth16x16Pixels?style=social)](https://github.com/Aveygo/WordsAreWorth16x16Pixels)   [![License: GPLv3](https://img.shields.io/badge/License-GPL-black.svg)](https://github.com/Aveygo/WordsAreWorth16x16Pixels/blob/main/license.txt) [![Python 3.9.9](https://img.shields.io/badge/python-3.9.9-black.svg)](https://www.python.org/downloads/release/python-399/)

An (admittedly amature) attempt to combine the strength of CNNs for sequence prediction though PACING - **P**atched b**A**sed **C**onvolut**I**o**N**al **G**enerator

<p align="center">
  <img src="https://github.com/Aveygo/WordsAreWorth16x16Pixels/raw/main/images/diagram.png" />
</p>

## The Background

[An image is worth 16x16 words](https://arxiv.org/abs/2010.11929) has shown us how transformers can be applied to classify images by converting
patches into "tokens" for said model to learn from. On the other spectrum, [ConvNext](https://arxiv.org/abs/2201.03545) has shown how classical
CNNs outperform transformers on the same task by translating similar model architextual improvements.

Currently, transformers reign over text/sequence tasks, but the idea is to augment the data in such a way to
allow CNNs and their recent progression to perform the same task in a similar manner to visual transformers (patches).

If successful, then the utilisation of CNNs could durastically reduce memory consumption, infernence times, and training requirements, due their more efficient use of parameters.

## The How

So, to convert a sequence of words (tokens) into an image, we:
1. Compute embeddings from tokens
2. Add positional encoding
3. Reshape embeddings into patches
4. Add padding where nessesary.
5. Combine patches into a final image

This image is then passed on to a CNN for training/inferenece.
Specifically, for this experiment, 1024 tokens with an embedding size of 768 are converted to an image of size 3,512,512.

## The Experiment

Please note, this experiment was designed acknowledging computing constants and should be scaled up for a 
more fair comparison.

Here, the task is to see if transformers or CNNs are better at predicting the next token in a sequence.
The dataset at hand is a token sequence of serveral shakespeare texts.

The tranformer is GPT-small (pos=1024, ctx=1024, embd=768, layers=12, heads=12)
The CNN is ConvNextv2-base (depths=[3, 3, 20, 3], dims=[128, 256, 512, 1024]) * Note the changed depth of 20 at position 2, this is for a 
more equal parameter count.

Both have their positional and word embeddings trained from scratch, with an vocabulary size of 50256.
Both have an extremely similar parameter count (124.44M vs 124.75M), however ConvNext is admittedly 0.2% larger.
Both were trained for one epochs (12120 samples), batch size of 2,  AdamW(lr=2e-4, weight_decay=0, betas=(0.9, 0.999), eps=1e-8), gradient clipping of 1, with a targeted minimum 1 hour training session.

Both models had their crossentropy loss logged on a validation set of 100 samples.

## The Results

### Plots of loss during training
![Test Loss](https://github.com/Aveygo/WordsAreWorth16x16Pixels/raw/main/images/testset.png)
![Train Loss](https://github.com/Aveygo/WordsAreWorth16x16Pixels/raw/main/images/trainset.png)

### Average of final 50 test-set loss values

*Lower is better*

PACING - 6.916 (-13.7%)

GPT2 - 8.015 (+15.9%)

### Average samples/second, GeForce RTX 3070 8GB

*Higher is better*

PACING - 2.275 (+51.8%)

GPT2 - 1.499 (-34.1%)

### Memory usage of model

PACING - 0.5GB

GPT2 - 0.55GB

### Sample inference
*top_k = 50, seed=43*

PACING
```
 what rich fair nothing thee d nothing love's

With did fair thy heart even of best with love thee form fromty byWhen I his beauty with me mine a. thy self oWhen thou: of,yTo then beautyed thee thou then behold.ilsted one be time: night's the you. ill love best have

 love me is this

 see and with on then I on'sTo me nothing live this mine wealth not on live by behold I form thee of mine did behold
```

GPT2
```
 health corrupt now other night look' deeds my that dece yous age theySh glad eyesost
 face divine dull so grace hours my tender keep'one grace lofty eyes, keep health they ares head; one now eyesred one so now me that which make her, eyes mine from hours present be express none see health I express'a dece presents those' tender keep they head none glad health look' dull sa not themselves dulls none dece now night other themselves keep they'presentSh
```

## Conclusion

Given a lower final test-set loss while being faster and more memory efficent, with an extremely similar parameter count, CNN's outperformed GPT-2 in this
token prediction task.

I hope to see more experimentation in this, with larger models, more data, and for longer periods of time, as I personally was not able to achieve the computational
requirements for a more conclusive result. I believe it shows promise nonetheless.
