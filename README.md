# Gaussian Processes for Modeling Fashion Brand DNA: Classification and Identity Fuzziness
### Anonymous

## Dataset summary
| Dataset   |  Train |  Val  |  Test |  Total |
|-----------|--------|-------|-------|--------|
| Luxury101 | 11,743 | 1,678 | 3,356 | 16,777 |
| Luxury24  | 7,340  | 1,043 | 2,070 | 10,453 |
| Brand5    | 700    | 100   | 200   | 1,000  |


## Download Dataset
### Luxury101 dataset
Download from Google Drive [Luxury101](https://drive.google.com/file/d/16MERWudbMn0iGfZivEAiP-gehh1yV-DS/view?usp=sharing)
Store the dataset to the [datasets/fashion_brands](datasets/fashion_brands)

<figure>
  <img src="figures/brand_sample1.png" alt="Sample images of luxury brands on our dataset." width="60%">
  <figcaption>Sample images of luxury brands on our dataset.</figcaption>
</figure>

### Luxury24 dataset
The Luxury24 dataset can be split from the JSON file of  [Luxury101](https://drive.google.com/file/d/16MERWudbMn0iGfZivEAiP-gehh1yV-DS/view?usp=sharing)

### Brand5 dataset
Download from Google Drive [Brand5](https://drive.google.com/file/d/17-gXTL9S9ugUQwHduZeerYeyRpH0qHDe/view?usp=sharing)

Store the dataset to the [datasets/fashion_brands_looks](datasets/fashion_brands_looks)

<figure>
  <img src="figures/brand_looks_sample.jpg" alt="Iconic looks." width="100%">
  <figcaption>Sample looks of Brand5 dataset.</figcaption>
</figure>

## Environment Installation
pip install -r requirements.txt


## Training GP
Run the file:  [fashion_gp/run_model.py](fashion_gp/run_model.py)

## Reproducing the Results

### (1) Reproduce Figure 3 in the paper
#### Follow  📄 [fig3_brand_fuzziness_visual.ipynb](fig3_brand_fuzziness_visual.ipynb): 
<figure>
  <img src="figures/brand_fuzziness_example.jpg" width="100%">
  <figcaption>How the judged probability and the uncertainty measures change as more observations are given to the model.</figcaption>
</figure>

### (2) Reproduce Figure 4, 6, 7 in the paper

#### Follow 📄 [fig4_6_7_brand_fuzziness.ipynb](fig4_6_7_brand_fuzziness.ipynb):  
<figure>
  <img src="figures/fig8.jpg" width="100%">
  <figcaption>The relationship between judged probability and episteme of positive labels using GPF.</figcaption>
</figure>

### (3) Other details coming soon
