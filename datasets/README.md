# Datasets

## Amazon Reviews
Link: https://www.kaggle.com/datasets/mexwell/amazon-reviews-multi

We transform and filter this dataset as follows:
- English docs
- Include all (5) sentiments/stars: 1, 2, 3, 4, 5
- Include top 15 products: apparel, beauty, kitchen, toy, sports, automotive, pet_products, pc, shoes, grocery, book, baby_product, jewelry, industrial_supplies, furniture
- Sample 500 docs per star

Stats:
- Train size: 2000
- Test size: 2500
- Original columns: stars, product_category, text

Generated text columns:
- stars-summary
- stars-keyphrases
- product_category-summary
- product_category-keyphrases


## Spotify 900k
Link: https://www.kaggle.com/datasets/devdope/900k-spotify

We transform and filter this dataset as follows:
- Include top 5 emotions: joy, sadness, anger, love, fear
- Include top 10 genres: folk, gospel, pop, country, comedy, hip hop, rock, jazz, classical, reggae
- Sample 500 docs per genre

Stats:
- Train size: 3000
- Test size: 5000
- Original columns: emotion, genre, text

Generated text columns:
- emotion-summary
- emotion-keyphrases
- genre-summary
- genre-keyphrases


## 20 Newsgroups

We transform this dataset as follows
- remove headers footers quotes
- remove linebreaks
- remove empty texts

Stats:
- Train size: 15076
- Test size: 3770
- Original columns: text, topic

Generated text columns:
- topic-summary
- topic-keyphrases