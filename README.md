# A word is worth 16x16 pixels

An (admittedly amature) attempt to combine the strengths of CNNs for sequence prediction.

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



## Conclusion

## TLDR

~Insert meme here~

