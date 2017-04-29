# 2016.M3.TQF-ML.HotelReviewSentiment
More detail see the pre_wangyu.pptx: 
https://github.com/renee-wangyu/2016.M3.TQF-ML.HotelReviewSentiment/blob/master/pre_wangyu.pptx

1. Project Description

This project will use a data set consisting of Chinese hotel reviews from www.ctrip.com to build sentiment classifiers that classifies a review as positive or negative.


2. Data

http://www.datatang.com/data/11936

ChnSentiCorp-Htl-ba-6000: balanced corpus，positive(3000 reviews) /negative（3000 reviews）

(since there are too many txt. files (6000 documents, each review is a document), I only uplaod some examples:

https://github.com/renee-wangyu/2016.M3.TQF-ML.HotelReviewSentiment/tree/master/data%20example


3. Process


· Chinese text segmentation

Chinese text segmentation（divide text into words) is an important question of Chinese information processing.

Using Jieba: https://github.com/fxsjy/jieba
（please scroll to bottom of the webpage, there are some explainations in English）

 A Hidden Markov Model (HMM)-based model is used with the Viterbi algorithm.


· Stop-word removal

Stop-words are simply those words that are extremely common in all sorts of texts and likely bear no (or only little) useful information that can be used to distinguish between different classes of reviews.

stoplis.txt: 
https://github.com/renee-wangyu/2016.M3.TQF-ML.HotelReviewSentiment/blob/master/code/stoplis.txt

After the prcessing, I have seg_pst.txt and seg_neg.txt.

https://github.com/renee-wangyu/2016.M3.TQF-ML.HotelReviewSentiment/blob/master/resluts/seg_neg.txt

https://github.com/renee-wangyu/2016.M3.TQF-ML.HotelReviewSentiment/blob/master/resluts/seg_pos.txt


· Bag-of-words model, Transforming words into feature vectors

Using bag-of-words (BOW) model to represent text as numerical feature vectors. 

Create a vocabulary of unique words—from the entire set of reviews dataset. I have dictionaries: 

https://github.com/renee-wangyu/2016.M3.TQF-ML.HotelReviewSentiment/blob/master/resluts/my_dict_pos.txt

https://github.com/renee-wangyu/2016.M3.TQF-ML.HotelReviewSentiment/blob/master/resluts/my_dict_neg.txt


· TFIDF (term frequency-inverse document frequency)

term frequency-inverse document frequency (tf-idf) that can be used to downweight those frequently occurring words in the feature vectors.


4. Data split

5. Classify methods: SVM, Decision tree, Logistics regression, Naive bayes

6. Prediction results

https://github.com/renee-wangyu/2016.M3.TQF-ML.HotelReviewSentiment/tree/master/resluts

7. Improvement


· Documentary frequency

If a word only shows in 1 or 2 reviews of all 6000 reviews, I think    the word is not important and remove it.
 (documentary frequency ≥ 3)


· Context: A triple model
Combine words in sequence and consider the context information

8. Conclusion

There is no significant difference between different methods, the possible reason is that the dataset is not large enough. 
Data processing/feature engineering is more important.
To improve the accuracy, we should have larger dataset.


Code: https://github.com/renee-wangyu/2016.M3.TQF-ML.HotelReviewSentiment/blob/master/code/wangyu.py

(there are some problems with my anaconda for mac, after pip installing jieba package( Chinese text segmentation package), I still can not use it with spyder and jupyter. I use Pycharm to do my project, and I only upload .py document. For your convenience, I save the results(several .txt documents) and pictures after run the code, you can check it in the results folder)

https://github.com/renee-wangyu/2016.M3.TQF-ML.HotelReviewSentiment/tree/master/resluts
