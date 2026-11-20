# GenLab: A Unified Generative AI Studio for Image Synthesis, Reconstruction and Evaluation

GenLab is a web application for experimenting with generative AI models. You pick a dataset, train a model in your browser, and then use it: reconstruct a drawing, remove noise, restyle a picture, generate faces, type a prompt, or caption an image. A second mode shows the template code beside the dataset so the same model can be studied and edited.

Everything runs locally in the browser with TensorFlow.js. There is no server, no Python, no GPU setup and no dataset download.

---

## 1. Quick start

1. Open `dist/genlab.html` in Chrome, Edge or Firefox (double-click is enough; works offline).
2. Choose a dataset in the top bar: **Digits**, **Shapes** or **Cartoon faces**.
3. Open a tool from the left menu, press **Train**, then use the *Try it* panels that appear.

Training speed depends on the browser backend (shown as a badge in the top bar). `webgl` (GPU) is fast; `cpu` is much slower. Use the **Quick** preset first.

To rebuild from source:

```
npm install @tensorflow/tfjs@4.22.0
node build.js        # writes dist/genlab.html
```

---

## 2. The two modes

Switch at any time with the toggle in the top bar.

### Easy Studio
For using models without reading code.
- Dataset card with a sample strip.
- Training length: **Quick / Balanced / Thorough**, plus optional advanced settings.
- One **Train** button, a live progress bar, loss curves and a gallery that updates while training.
- **Try it** panels: draw or upload an image, move latent sliders, type a prompt, and so on.
- A collapsible "How does it work?" section with the key equation and suggested experiments.

### Learn Lab
For understanding models. The screen is split side by side.
- **Left:** the model's template code, with labelled steps (for example "STEP 3 · Define the loss") that you can click to jump to. A settings strip sits above it. Edit anything and press **Run** (or Ctrl+Enter). **Reset code** restores the original.
- **Right, in tabs:**
  - **Data:** the dataset grid. Click an image to see its raw 16×16 pixel numbers.
  - **Output:** live loss curves, image galleries and a console with errors and the line number.
  - **Try it:** the same interactive panel as Easy mode, using the model you just trained or edited.
  - **Concept:** short explanation, the main formula, and things to try.

The code you see is the exact code that runs. Easy mode runs the same scripts without showing them.

---

## 3. Tools

| Tool | What you can do | Syllabus topic |
|---|---|---|
| **Autoencoder** | Compress and rebuild images; change code size; draw or upload your own. Overfit demo with dropout, L2 and training-set size; train vs validation curves. | Autoencoders, Regularization |
| **Denoiser** | Train on damaged images (static, missing pixels, salt and pepper) and repair new ones. | Denoising autoencoders |
| **VAE · Latent Space** | Generate new images; β control; live ELBO terms (reconstruction vs KL). Explore with sliders, a 2-D map, interpolation between two images, and variations of one image. | VAE and ELBO, latent exploration |
| **GAN · WGAN · DCGAN** | MLP or convolutional generator, vanilla or Wasserstein loss, learning timeline, noise-space explorer, **mode-collapse check**. | GAN, DCGAN, WGAN, mode collapse |
| **CycleGAN Style Transfer** | Translate between unpaired domains (negative, sketch, poster, colour swap). Shows cycle reconstruction and error against the known true style. | CycleGAN |
| **Diffusion Lab** | Class-conditional DDPM with classifier-free guidance; watch noise turn into an image across timesteps. | Diffusion models |
| **Text → Image** | Type a prompt such as "a smiling blond person"; the diffusion model follows it. Shares the model with Diffusion Lab. | Text prompts, diffusion |
| **RealNVP Flow** | Affine coupling flow; exact likelihood in bits per dimension; image → noise → image exact round trip. | RealNVP |
| **PixelRNN** | GRU that draws an 8×8 black-and-white image pixel by pixel, and completes half-finished images. | PixelRNN |
| **Transformer Captions** | A small GPT-style decoder with causal self-attention that writes captions for images; temperature control. | Transformers, GPT |
| **Compare & Metrics** | Scores all trained generators with FID, Inception Score, diversity and class coverage; flags possible mode collapse. | FID and IS |
| **Project Map** | Syllabus topic → where it lives in the app. | |

### Suggested path for a first session
1. **Autoencoder** (Learn Lab): read the five steps, run, then change the code size from 16 to 2.
2. **VAE**: set latent size 2 and open the Map tab.
3. **GAN**: train vanilla GAN, then WGAN. Press *Check diversity* each time.
4. **Compare & Metrics**: *Quick-train 5 generators*, then *Run evaluation*.
5. **Diffusion** and **Text → Image** on Cartoon faces.

---

## 4. Datasets

Datasets are drawn procedurally on a canvas when the app starts, so they are small, reproducible and need no download. All images are 16×16 pixels with values from 0 to 1.

| Dataset | Channels | Classes | Notes |
|---|---|---|---|
| Digits | 1 | 10 | Digits 0-9 in several fonts, with tilt, shear and thickness changes. |
| Shapes | 1 | 6 | Circle, square, triangle, cross, ring, diamond at random size, position and rotation. |
| Cartoon faces | 3 | 8 | 4 hair colours × smiling or frowning, with random skin tones and backgrounds. |

Each set has 2000 training and 400 test images, plus a short caption per image used by the captioner.

---

## 5. Evaluation and metrics

The Compare tab samples 300 images from every generator you have trained on the current dataset and reports:

- **FID** (lower is better): Fréchet distance between feature statistics of real and generated images.
- **IS** (higher is better): Inception Score; high when images are confidently classified and cover many classes.
- **Diversity:** typical pixel distance between two samples, relative to real data.
- **Classes and entropy:** how many classes the generator covers. Low coverage means **mode collapse**.

A "Real data" row (test set vs a slice of the training set) is shown as a reference. In Learn Lab the Compare tab displays the metrics source code, which is the code that actually runs.

---

## 6. Project structure

```
build.js                 bundles everything into one HTML file
src/
  index.html             page template
  style.css              design tokens, light and dark themes, layout
  core.js                datasets, captions, prompt parser, FID/IS/diversity, evaluator, Runner
  tools.js               tool list, settings, presets, teaching text
  ui_a.js                code editor, charts output, dataset inspector, draw pad, latent explorer
  ui_b.js                Try-it panels, Compare page, Home, Project Map, app shell
  recipes/
    ae.js dae.js vae.js gan.js cyclegan.js diffusion.js realnvp.js pixelrnn.js captioner.js
dist/genlab.html         built, self-contained application
```

### How it works
- A **recipe** is a plain script with the variables `tf`, `data`, `P` (settings) and `lab` (helpers) in scope. It builds a model, trains it, calls `lab.plot` and `lab.show` to stream curves and images, and returns an object the Try-it panel uses.
- The **Runner** executes the recipe text, stores charts, galleries and the trained model, and registers generators so Compare can score them.
- Learn Lab edits the recipe text; Easy mode runs it unchanged.

### Adding a new model
1. Write `src/recipes/mymodel.js` following the existing ones.
2. Add an entry in `src/tools.js` (name, settings, presets, concept text).
3. Add a `TRYIT.mymodel` panel in `src/ui_b.js` and list it in `NAV`.
4. Optional: call `lab.register({id, label, sample})` so it appears in Compare.
5. Run `node build.js`.

---

## 7. Limitations

State these in a report so the results are read correctly.
- Images are 16×16 and models are small so they train in a browser tab. This is a teaching tool, not a photo generator. Undertrained models (Quick preset, or any run on a CPU backend) give noisy samples.
- FID and IS use the hidden layer of a small classifier trained on the current dataset, not Inception-v3. The formulas are the standard ones, but numbers are not comparable with published FID values. With 300 samples the scores are noisy.
- The text encoder is a keyword parser mapping words to dataset classes. It is not a transformer text model. The transformer in the app is the captioner.
- PixelRNN runs on 8×8 black-and-white versions of the images.
- CycleGAN's second domain is made from the data with known transforms, so quality can be measured, but it is not a real artistic-style dataset.
- Training runs in the browser: closing the tab loses trained models. Nothing is saved.

---

## 8. Requirements and compatibility

- A current desktop browser (Chrome, Edge, Firefox, Safari). WebGL is used when available.
- Works on a phone layout, but training is slow there.
- Web fonts load from Google Fonts when online; otherwise system fonts are used.
- Built with TensorFlow.js 4.22.0, bundled into the HTML file.

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| Training is very slow | Check the backend badge. If it says `cpu`, use Quick, lower the steps, or try another browser with hardware acceleration on. |
| Samples look like noise | Train longer (Balanced or Thorough). Diffusion and DCGAN need the most steps. |
| "Train at least one generator first" | Train a VAE, GAN, Diffusion or RealNVP model on the current dataset. Models are stored per dataset. |
| Error in Learn Lab console | The message shows the line number in your code. Press **Reset code** to restore the original. |
| Try-it panel says it could not be built | Your edited code must still return the same object the original returns. |
