# Fine-Tuned CLIP for Fashion Text-to-Image Retrieval

<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/094f72fc-d842-4298-b711-9f7b90533cc4" />


A reproducible CLIP fine-tuning and text-to-image product retrieval project built on a fashion catalogue of **44,160** image–description pairs. The project fine-tunes [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32), caches normalized image embeddings, and evaluates strict image-level retrieval on a validation split.


## What this project does

The notebook implements an end-to-end retrieval pipeline. It inspects and cleans catalogue metadata, fine-tunes CLIP with its standard symmetric contrastive loss, selects the checkpoint with the highest observed validation diagonal logit, builds a cache of image embeddings for the complete catalogue, and retrieves the top-ranked images for a new English-language query.

The experiment is designed as a **decision-support prototype for product discovery**, not as an autonomous recommendation or ranking system.

## Results from the recorded reproducible run

The run uses seed `42`, **39,744** training pairs, and **4,416** validation pairs. The values below are the complete validation split results, not results from a single batch.

| Measure | Result |
|---|---:|
| Full-validation diagonal CLIP logit before fine-tuning | 29.87 |
| Best full-validation diagonal CLIP logit | **31.34** |
| Absolute change | **+1.47** |
| Relative change | **+4.92%** |
| Selected checkpoint | `clip_best.pt` from epoch 4 |
| Recall@1 | 0.0876 |
| Recall@5 | 0.2620 |
| Recall@10 | 0.3698 |
| Mean Reciprocal Rank (MRR) | 0.1809 |

The course target, a validation diagonal CLIP logit above 30, was exceeded from epoch 2 onward. The final selected checkpoint was produced at epoch 4.

## Retrieval metrics

Strict image-level retrieval is evaluated with each validation description as a query against the full **44,160-image** gallery. Each query has one exact paired image filename treated as relevant.

- **Recall@1** is the proportion of queries whose paired image is ranked first.
- **Recall@5** and **Recall@10** measure whether that image appears among the first five or ten results.
- **MRR** gives higher credit when the paired image appears closer to the top of the ranking.

These metrics evaluate the retrieval task directly. The diagonal CLIP logit is retained because it was the course fine-tuning criterion, but it is not a probability or a retrieval metric.

## Repository structure

```text
.
├── clip_fashion_text_to_image_retrieval.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

## Data

The source data is **not included** in this repository. It contains a large image catalogue and should not be committed to GitHub. The notebook expects a local Google Drive archive at:

```text
/content/drive/MyDrive/archive.zip
```

When extracted, the archive must contain the following layout:

```text
/content/clothing_data/
├── data.csv
└── data/
    ├── 1.jpg
    ├── 2.jpg
    └── ...
```

The work is based on the public [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset), which Kaggle describes as a catalogue of approximately 44k products with category labels, descriptions, and high-resolution images. The recorded notebook uses a course-provided archive with a simplified `data.csv` plus `data/` layout. Download and use the source data under its own terms; do not upload the extracted catalogue, checkpoints, or embedding cache to this repository. [1]

## Running the notebook in Google Colab

1. Download or prepare the dataset archive with the expected layout above and upload it privately to Google Drive as `MyDrive/archive.zip`.
2. Open `notebooks/clip_fashion_text_to_image_retrieval.ipynb` in Google Colab.
3. Select **Runtime → Change runtime type → T4 GPU** (or another available GPU).
4. Run the notebook from top to bottom and authorize Google Drive when prompted.
5. The notebook saves checkpoints to `MyDrive/clip_checkpoints/` and embedding-cache files to `MyDrive/clip_search_index/`.

For a local environment, install dependencies with:

```bash
pip install -r requirements.txt
```

The notebook was written for Google Colab because its input archive, checkpoints, and embedding cache are stored in Google Drive.

## Reproducibility and implementation choices

- Python, NumPy, PyTorch, CUDA, DataLoader generator, and worker random states are seeded with `42`.
- Text is tokenized with `truncation=True` in both training and query-time retrieval.
- Image and text vectors are L2-normalized, so their dot product is cosine similarity.
- `clip_best.pt` is selected by the highest full-validation mean diagonal CLIP logit.
- Image embeddings are calculated once for the complete catalogue and cached, so a new query requires only text encoding and vector similarity search.
- Tensor-only checkpoints are loaded with `weights_only=True`.

## Limitations

The recorded 90/10 split is a reproducible **validation split**, not an untouched test split: it is observed after every training epoch and used to select the checkpoint. Therefore the reported metrics are not independent final test results.

The catalogue contains 5,304 repeated descriptions attached to different images. The strict metric treats only one exact image filename as relevant, which can understate semantic relevance when multiple images share the same description. Conversely, a random image-level split can place identical descriptions in train and validation, making results optimistic for entirely unseen descriptions.

The notebook does not compute Recall@K or MRR for the original pretrained CLIP model. It supports a positive result under the course diagonal-logit criterion and reports direct retrieval quality for the fine-tuned model, but it does not establish a retrieval-metric improvement over the base model.

[1]: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset "Fashion Product Images Dataset on Kaggle"
[2]: https://huggingface.co/openai/clip-vit-base-patch32 "OpenAI CLIP ViT-B/32 model card"
[3]: https://huggingface.co/docs/transformers/model_doc/clip "Hugging Face Transformers CLIP documentation"
[4]: https://docs.pytorch.org/docs/2.14/notes/randomness.html "PyTorch reproducibility guidance"

